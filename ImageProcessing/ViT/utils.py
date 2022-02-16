# -*- coding: utf-8 -*-
# @Time : 2022/2/16 12:10
# @Author : Alethia Chaos
# @Email : 2019141620371@stu.scu.edu.cn
# @File : utils.py
# @Project : deep-learning
from torch import Tensor
import torch.nn as nn


def drop_path(x: Tensor, drop_prob=0., training=False, scale_by_keep=True):
    """
    实现随机丢弃残差层
    :param x: input Tensor
    :param drop_prob: float default = 0.
    :param training: bool default = False
    :param scale_by_keep: bool default = True
    :return:
    """
    if drop_prob == 0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # shape : (batch, 1 ,1 ,1) (将输入除第一维之外变成1)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    # 生成b_l伯努利随机变量 表明此层是激活还是失活 keep_prob 为保存比例
    random_tensor = x.new_empty(shape).bernoulli(keep_prob)
    # scale_by_keep 根据保留概率放大原输入
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """
    实现Stochastic Drop
    """

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None, flatten=True):
        super(PatchEmbed, self).__init__()
        self.flatten = flatten
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm_layer = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H} * {W}) dosen't match model ({self.img_size[0]} * {self.img_size[1]})"
        # flatten: [B, C, H, W] -> [B, C, H*W]
        # transpose: [B, C, H*W] -> [B, H*W, C]
        # 此处的C已经由proj层变成了embed_dim大小 H = grid_size[0] W = grid_size[1]
        # H*W为序列长度 C为向量特征数量
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm_layer(x)
        return x
