# -*- coding: utf-8 -*-
# @Time : 2022/2/13 11:51
# @Author : Alethia Chaos
# @Email : 2019141620371@stu.scu.edu.cn
# @File : SublayerConnection.py
# @Project : Transformer
import torch.nn as nn
from torch.nn import LayerNorm, Dropout


class SublayerConnection(nn.Module):
    """
    实现的是LayerNorm + Sublayer(Attention/Dense) + dropout + Residual Connection
    与原论文不同在于将LayerNorm放在了前面 原论文是将LayerNorm放在残差连接之后
    """

    def __init__(self, size, dropout_rate):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = Dropout(dropout_rate)

    def forward(self, x, sublayer):
        """

        :param x: 输入
        :param sublayer: 可能是Attention或者是全连接模块 输入为一个参数
        :return: x + dropout(sublayer(layernorm(x)))
        """
        return x + self.dropout(sublayer(self.norm(x)))
