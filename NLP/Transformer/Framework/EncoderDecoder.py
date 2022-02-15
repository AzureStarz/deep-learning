# -*- coding: utf-8 -*-
# @Time : 2022/2/13 10:51
# @Author : Alethia Chaos
# @Email : 2019141620371@stu.scu.edu.cn
# @File : EncoderDecoder.py
# @Project : Transformer
import torch.nn as nn


class EncoderDecoder(nn.Module):
    """
    标准的Encoder-Decoder架构，是很多模型的基础
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        # encoder和decoder都在构造的时候传入，这样会非常灵活
        self.encoder = encoder
        self.decoder = decoder
        # 目标语言与源语言的embedding层
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        # 根据Decoder的隐状态输出当前时刻的词
        # 实现： Linear + softmax
        # Linear输出个数是词的个数， 之后跟个softmax
        # 输出概率最高的词
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        # 首先调用encode方法对输入进行编码，然后调用decode方法解码
        """

        :param src: 源语言（输入）
        :param tgt: 目标语言（输出）
        :param src_mask: 源语言的掩码
        :param tgt_mask: 目标语言掩码
        :return: forward propagation
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        """

        :param src: 源语言（输入）
        :param src_mask: 源语言的掩码
        :return: 输入编码结果
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """

        :param memory: 前i个输出结果（解码器只能逐个逐个看 利用前i个的输出结果来进行解码）
        :param src_mask: 源语言的掩码
        :param tgt: 目标语言（输出）
        :param tgt_mask: 目标语言掩码
        :return: 解码结果
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
