# -*- coding: utf-8 -*-
# @Time : 2022/2/13 10:58
# @Author : Alethia Chaos
# @Email : 2019141620371@stu.scu.edu.cn
# @File : Generator.py
# @Project : Transformer
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """
    根据Decoder的隐状态输出当前时刻的词
    实现： Linear + softmax
    Linear输出个数是词的个数， 之后跟个softmax
    输出概率最高的词
    """
    def __init__(self, d_model, vocab):
        """

        :param d_model: Decoder输出的大小
        :param vocab: 词典大小
        """
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    # 全连接再加上一个softmax
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=1)
