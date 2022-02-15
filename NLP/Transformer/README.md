# Transformer 代码实现

## Overall Framework
-  EncoderDecoder.py : 标准的EncoderDecoder架构，是许多模型的基础；可以复用到别的同样是编码器解码器架构中;将编码器的输出作为解码器解码的输入之一(memory)
-  Encoder : 将N个Encoder Layer stack起来 形成Encoder
-  Decoder : 将N个Decoder Layer stack起来 行程Decoder
-  Layer : 由 Sublayer Connection构成  Sublayer又是Attention或者PositionWisedFeedForward(MLP)
-  Generator : 将解码器最后的输出接上Linear Softmax转变成概率
-  Utils里面就是需要用到的一些方法 Attention MultiHeadedAttention SubsequentMask Embedding Positional Encoding等都在这里面实现

## Remark
具体的数据集和训练就没有实现，只是复现了Transformer结构的细节

## References
[The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

[Transformer代码阅读](http://fancyerii.github.io/2019/03/09/transformer-codes/)

