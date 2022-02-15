# -*- coding: utf-8 -*-
# @Time : 2022/2/13 17:38
# @Author : Alethia Chaos
# @Email : 2019141620371@stu.scu.edu.cn
# @File : main.py
# @Project : Transformer
import copy
from Decoder.DecodeLayer import DecodeLayer
from Decoder.Decoder import Decoder
from Encoder.Encoder import Encoder
from Framework.EncoderDecoder import EncoderDecoder
from Encoder.EncoderLayer import EncoderLayer
from Framework.Generator import Generator
from Utils.Utils import MultiHeadedAttention, PositionwiseFeedForward, PositionalEncoding, Embeddings
import torch.nn as nn


def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, dropout)
    position_encoding = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecodeLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position_encoding)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position_encoding)),
        Generator(d_model, tgt_vocab)
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
