# -*- coding: utf-8 -*-
# @Time : 2022/2/16 11:45
# @Author : Alethia Chaos
# @Email : 2019141620371@stu.scu.edu.cn
# @File : vit_model.py
# @Project : deep-learning
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn

from ImageProcessing.ViT.utils import DropPath, PatchEmbed


class Attention(nn.Module):
    """
    ViT中的Attention都是self-attention QKV由输入经过线性变换得到的
    """

    def __init__(self, total_dim, num_heads=8, attn_drop=0., proj_drop=0., qkv_bias=False):
        super(Attention, self).__init__()
        assert total_dim % num_heads == 0, "total_dim should be divisible by num_heads"
        self.embed_dim_per_head = total_dim // num_heads
        self.total_dim = total_dim
        self.num_head = num_heads
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.qkv_bias = qkv_bias
        self.scale = self.embed_dim_per_head ** -0.5
        # 构造QKV线性变换层
        self.qkv = nn.Linear(total_dim, total_dim * 3)
        self.proj = nn.Linear(total_dim, total_dim)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape
        # [batch_size, num_patches + 1, 3 * total_embed_dim]
        qkv = self.qkv(x)
        # [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = qkv.reshape(B, N, 3, self.num_head, self.embed_dim_per_head).permute(2, 0, 3, 1, 4)
        # unbind(0)将tensor的第一维拆出来 此处为拆成Q K V
        q, k, v = qkv.unbind(0)

        # transpose: [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @ multiply: [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @ multiply: [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FeedForward(nn.Module):
    """
    由MLP实现
    """

    def __init__(self, in_features, act_layer=nn.GELU, scale_ratio=4., drop_ratio=0.):
        super(FeedForward, self).__init__()
        hidden_features = in_features * scale_ratio
        out_features = in_features
        self.dense1 = nn.Linear(in_features, hidden_features)
        self.dense2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, x):
        x = self.dense2(self.dropout(self.act(self.dense1(x))))
        return self.dropout(x)


class Block(nn.Module):
    """
    ViT的一个Block 由两层Sublayer组成 分别是 Attention FeedForward
    """

    def __init__(self, total_dim, num_heads, attn_drop=0., drop_ratio=0., proj_drop=0., drop_path=0.,
                 scale_ratio=4., qkv_bias=False, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        # 1 sublayer Attention
        self.norm1 = norm_layer(total_dim)
        self.attn = Attention(total_dim, num_heads, attn_drop, proj_drop, qkv_bias)
        # 2 sublayer MLP
        self.norm2 = norm_layer(total_dim)
        self.feed_forward = FeedForward(total_dim, act_layer, scale_ratio, drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.feed_forward(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                total_dim=embed_dim, num_heads=num_heads, scale_ratio=mlp_ratio, qkv_bias=qkv_bias,
                attn_drop=attn_drop_rate, drop_ratio=drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation Layer 相当于多了一层Linear
        if representation_size > 0:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier Heads
        self.head = nn.Linear(self.num_features, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        # expand: [batch_size, 1, embed_dim]  tips: -1 means not changing the size of that dimension
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((x, cls_token), dim=1)
        x = self.pos_drop(x=self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        # x[:, 0]是将cls单独切出来做最后的全连接
        return self.pre_logits(x[:, 0])

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
