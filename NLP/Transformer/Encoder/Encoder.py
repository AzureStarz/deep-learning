# -*- coding: utf-8 -*-
# @Time : 2022/2/13 11:10
# @Author : Alethia Chaos
# @Email : 2019141620371@stu.scu.edu.cn
# @File : Encoder.py
# @Project : Transformer
from torch.nn import LayerNorm
from Utils.Utils import clones
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder是将N个EncoderLayer串起来的结构
    这里的Encoder与原论文中Encoder的LayerNorm顺序不太一样
    原论文是
        x -> self-attention(x) -> x + self-attention(x) -> layernorm(x + self-attention(x)) => y
        y -> dense(y) -> y + dense(y) -> layernorm(y + dense(y)) => z(输入到下一层)
    现在是
        x -> layernorm(x) -> self-attention(layernorm(x)) -> x + self-attention(layernorm(x)) => y
        y -> layernorm(y) -> self-attention(layernorm(y)) -> y + self-attention(layernorm(y)) => z
    """

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        # layer是一个EncoderLayer layers是将N个完全相同的EncoderLayer串起来
        self.layers = clones(layer, N)
        # LayerNorm层 （比BatchNorm效果好）
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        # 最后进行LayerNorm ？为什么最后还有个LayerNorm
        return self.norm(x)
