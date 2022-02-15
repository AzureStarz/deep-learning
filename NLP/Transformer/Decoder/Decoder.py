# -*- coding: utf-8 -*-
# @Time : 2022/2/13 16:03
# @Author : Alethia Chaos
# @Email : 2019141620371@stu.scu.edu.cn
# @File : Decoder.py
# @Project : Transformer
import torch.nn as nn
from torch.nn import LayerNorm

from Utils.Utils import clones


class Decoder(nn.Module):
    """
    整体解码器的结构
    """

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
