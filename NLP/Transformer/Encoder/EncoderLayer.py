# -*- coding: utf-8 -*-
# @Time : 2022/2/13 15:34
# @Author : Alethia Chaos
# @Email : 2019141620371@stu.scu.edu.cn
# @File : EncoderLayer.py
# @Project : Transformer
import torch.nn as nn
from Framework.SublayerConnection import SublayerConnection
from Utils.Utils import clones


class EncoderLayer(nn.Module):
    """
    一层EncoderLayer包括 self-attention层 及 基于位置的前馈神经网络（dense）
    """

    def __init__(self, size, self_attn, feed_forward, dropout_rate):
        super(EncoderLayer, self).__init__()
        # attention是dot attention计算
        self.self_attn = self_attn
        # 基于位置的前馈神经网络
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout_rate), 2)
        self.size = size

    def forward(self, x, mask):
        # SublayerConnection的forward参数中的sublayer需要callable
        # 所以self_attn需要用lambda来创建可调用对象(即函数)
        # self.sublayer[0]是个callable，self.sublayer[0](x, z)会调用self.sublayer[0].call(x, z)
        # 然后会调用SublayerConnection.forward(x, z)，然后会调用sublayer(self.norm(x))
        # sublayer就是传入的参数z，因此就是z(self.norm(x))
        # z是一个lambda，我们可以先简单的看成一个函数，显然这里要求函数z的输入是一个参数。
        x = self.sublayer[0](x, lambda y: self.self_attn(y, y, y, mask))
        return self.sublayer[1](x, self.feed_forward)
