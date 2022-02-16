# -*- coding: utf-8 -*-
# @Time : 2022/2/13 11:31
# @Author : Alethia Chaos
# @Email : 2019141620371@stu.scu.edu.cn
# @File : utils.py
# @Project : Transformer
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from torch.autograd import Variable


def clones(module, N):
    # 克隆N个完全相同的子层 利用copy.deepcopy
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def sub_sequent_mask(size):
    """
    生成遮罩，使得attention只能使用前t个时刻的输入
    :param size: 总共多少个输入
    :return: 一个下三角矩阵
    """
    # 遮罩是 1， size， size 大小的矩阵
    # 由np.triu生成上三角矩阵
    # 由torch.from_numpy 从numpy的矩阵生成tensor 并取下三角矩阵 （通过 == 0 操作）
    # 这样就生成了下三角为1 的遮罩
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    # query key 的shape都是(num_batch, num_heads, seq_length, num_feature)
    d_k = query.shape[-1]
    # matmul会把query和key的最后两维进行矩阵乘法，这样效率更高
    # attention_scores 的shape是(num_batch, num_heads, seq_length, seq_length)
    # a_ij 代表时刻i attention 时刻j 的注意力分数
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
    probability_attention_scores = F.softmax(attention_scores, dim=-1)
    if dropout is not None:
        probability_attention_scores = dropout(probability_attention_scores)
    return torch.matmul(probability_attention_scores, value), probability_attention_scores


class MultiHeadedAttention(nn.Module):
    """
    多头注意力
    """
    def __init__(self, h, d_model, dropout=0.1):
        """

        :param h: head的个数
        :param d_model: MultiHead输出大小
        :param dropout:
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        # Q K V 对应的线性变换 W_Q, W_K, W_V
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        # concat之后Multihead的总输出的线性变换 用于输出降维（此处因为刚好8个头对应的d_k × h和输出一样所以都是d_model）
        self.last_linear = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # 所有h个head的mask都是相同的
            mask = mask.unsqueeze(1)
        n_batch = query.size(0)

        # 1) 先对Q K V使用线性变换
        query, key, value = [f(x).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
                             for f, x in zip(self.linears, (query, key, value))]

        # 2) attention函数计算
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) 把h个head生成的d_k个向量concat起来，再使用linear
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.d_k*self.h)
        return self.last_linear(x)


class PositionwiseFeedForward(nn.Module):
    """
    全连接层有两个线性变换以及它们之间的ReLU激活组成
    全连接层的输入和输出都是d_model(512)维的，中间隐单元的个数是d_ff(2048)
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __int__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
