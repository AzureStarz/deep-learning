# -*- coding: utf-8 -*-
# @Time : 2022/2/13 16:06
# @Author : Alethia Chaos
# @Email : 2019141620371@stu.scu.edu.cn
# @File : DecodeLayer.py
# @Project : Transformer
from Framework.SublayerConnection import SublayerConnection
import torch.nn as nn
from Utils.Utils import clones


class DecodeLayer(nn.Module):
    """
    Decoder与Encoder关键不同在于Decoder在解码第t个时刻的时候只能使用1...t时刻的输入，而不能使用t+1时刻及之后的输入
    """

    def __init__(self, size, dropout_rate, feed_forward, self_attn, src_attn):
        super(DecodeLayer, self).__init__()
        self.feed_forward = feed_forward
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.sublayer = clones(SublayerConnection(size, dropout_rate), 3)
        self.size = size

    def forward(self, x, memory, tgt_mask, src_mask):
        # self_attn 与 src_attn 实现是一样的 都是dot attention
        # self_attn 在于 Q K V 都是来自输入
        # src_attn Q是来自Decoder的上层输入 （K V） 是来自Encoder的输出 即 :param memory
        x = self.sublayer[0](x, lambda y: self.self_attn(y, y, y, tgt_mask))
        x = self.sublayer[1](x, lambda y: self.src_attn(y, memory, memory, src_mask))
        return self.sublayer[2](x, self.feed_forward)

