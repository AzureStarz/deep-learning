from itertools import repeat

import torch
import torch.nn as nn
from torch.nn import LayerNorm


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
        Args:
            img_size (int): Image size.  Default: 224.
            patch_size (int): Patch token size. Default: 4.
            in_chans (int): Number of input image channels. Default: 3.
            embed_dim (int): Number of linear projection output channels. Default: 96.
            norm_layer (nn.Module, optional): Normalization layer. Default: None
        """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=3, norm_layer=None):
        super(PatchEmbed, self).__init__()
        img_size = tuple(repeat(img_size, 2))
        patch_size = tuple(repeat(patch_size, 2))
        patches_resolution = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.embed_dim = embed_dim
        self.in_chans = in_chans

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})"
        x = self.proj(x).flatten(2).transpose(1, 2)  # 变成 B C H*W 再变成 B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=LayerNorm):
        """

        Args:
            input_resolution (tuple[int]): Resolution of input feature.
            dim (int): Number of input channels.
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        """
        super(PatchMerging, self).__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        self.input_resolution = input_resolution

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even"
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H*W/4 4*C

        x = self.norm(x)
        x = self.reduction(x)  # [B, H*W/4, 4*C]

        return x


def window_partition(x, window_size):
    """

    Args:
        x (): [B, H, W, C]
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 3, 1, 2, 4).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """

    Args:
        windows (): [B*num_windows, window_size, window_size, C]
        window_size (): window size
        H (): Height of image
        W (): Width of image

    Returns:
        x : [B, H, W, C]
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1);
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qkv_scale=None, attn_drop=0., proj_drop=0.):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qkv_bias or head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None

        Returns:

        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv_proj -> [B_, N, 3, num_heads, C // num_heads]
        # permute -> [3, B_, num_heads, N, C // num_heads]
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = (q @ k.transpose(1, 2))

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


