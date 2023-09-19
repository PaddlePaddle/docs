.. _cn_api_paddle_nn_Transformer:

Transformer
-------------------------------

.. py:class:: paddle.nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', attn_dropout=None, act_dropout=None, normalize_before=False, weight_attr=None, bias_attr=None, custom_encoder=None, custom_decoder=None)



**Transformer 模型**

Transformer 模型由一个 ``TransformerEncoder`` 实例和一个 ``TransformerDecoder`` 实例组成，不包含 embedding 层和输出层。

细节可参考论文 `Attention is all you need <https://arxiv.org/pdf/1706.03762.pdf>`_ 。

用户可以使用相应的参数配置模型结构。请注意 ``normalize_before`` 的用法与某些类似 Transformer 的模型例如 BERT 和 GPT2 的用法不同，它表示在哪里（多头注意力机制或前馈神经网络的输入还是输出）进行层标准化（Layer Normalization）。该模型默认的结构是对每个子层的 output 进行层归一化，并在最后一个编码器/解码器的输出上进行另一个层归一化操作。


参数
::::::::::::

    - **d_model** (int，可选) - 编码器和解码器的输入输出的维度。默认值：512。
    - **nhead** (int，可选) - 多头注意力机制的 Head 数量。默认值：8。
    - **num_encoder_layers** (int，可选) - 编码器中 ``TransformerEncoderLayer`` 的层数。默认值：6。
    - **num_decoder_layers** (int，可选) - 解码器中 ``TransformerDecoderLayer`` 的层数。默认值：6。
    - **dim_feedforward** (int，可选) - 前馈神经网络中隐藏层的大小。默认值：2048。
    - **dropout** (float，可选) - 对编码器和解码器中每个子层的输出进行处理的 dropout 值。默认值：0.1。
    - **activation** (str，可选) - 前馈神经网络的激活函数。默认值：``relu``。
    - **attn_dropout** (float，可选) - 多头自注意力机制中对注意力目标的随机失活率。如果为 ``None`` 则 ``attn_dropout = dropout``。默认值：``None``。
    - **act_dropout** (float，可选) - 前馈神经网络的激活函数后的 dropout。如果为 ``None`` 则 ``act_dropout = dropout``。默认值：``None``。
    - **normalize_before** (bool，可选) - 设置对编码器解码器的每个子层的输入输出的处理。如果为 ``True``，则对每个子层的输入进行层标准化（Layer Normalization），对每个子层的输出进行 dropout 和残差连接（residual connection）。否则（即为 ``False``），则对每个子层的输入不进行处理，只对每个子层的输出进行 dropout、残差连接（residual connection）和层标准化（Layer Normalization）。默认值：``False``。
    - **weight_attr** (ParamAttr|tuple，可选) - 指定权重参数属性的对象。如果是 ``tuple``，则只支持 ``tuple`` 长度为 1、2 或 3 的情况。如果 ``tuple`` 长度为 3，多头自注意力机制的权重参数属性使用 ``weight_attr[0]``，解码器的编码-解码交叉注意力机制的权重参数属性使用 ``weight_attr[1]``，前馈神经网络的权重参数属性使用 ``weight_attr[2]``；如果 ``tuple`` 的长度是 2，多头自注意力机制和解码器的编码-解码交叉注意力机制的权重参数属性使用 ``weight_attr[0]``，前馈神经网络的权重参数属性使用 ``weight_attr[1]``；如果 ``tuple`` 的长度是 1，多头自注意力机制、解码器的编码-解码交叉注意力机制和前馈神经网络的权重参数属性都使用 ``weight_attr[0]``。如果该参数值是 ``ParamAttr``，则多头自注意力机制、解码器的编码-解码交叉注意力机制和前馈神经网络的权重参数属性都使用 ``ParamAttr``。默认值：``None``，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
    - **bias_attr** （ParamAttr|tuple|bool，可选）- 指定偏置参数属性的对象。如果是 ``tuple``，则只支持 ``tuple`` 长度为 1、2 或 3 的情况。如果 ``tuple`` 长度为 3，多头自注意力机制的偏置参数属性使用 ``bias_attr[0]``，解码器的编码-解码交叉注意力机制的偏置参数属性使用 ``bias_attr[1]``，前馈神经网络的偏置参数属性使用 ``bias_attr[2]``；如果 ``tuple`` 的长度是 2，多头自注意力机制和解码器的编码-解码交叉注意力机制的偏置参数属性使用 ``bias_attr[0]``，前馈神经网络的偏置参数属性使用 ``bias_attr[1]``；如果 ``tuple`` 的长度是 1，多头自注意力机制、解码器的编码-解码交叉注意力机制和前馈神经网络的偏置参数属性都使用 ``bias_attr[0]``。如果该参数值是 ``ParamAttr``，则多头自注意力机制、解码器的编码-解码交叉注意力机制和前馈神经网络的偏置参数属性都使用 ``ParamAttr``。如果该参数为 ``bool`` 类型，只支持为 ``False``，表示没有偏置参数。默认值：``None``，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
    - **custom_encoder** (Layer，可选) - 若提供该参数，则将 ``custom_encoder`` 作为编码器。默认值：``None``。
    - **custom_decoder** (Layer，可选) - 若提供该参数，则将 ``custom_decoder`` 作为解码器。默认值：``None``。


代码示例
::::::::::::

COPY-FROM: paddle.nn.Transformer

方法
::::::::::::
forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None)
'''''''''

将 Transformer 应用于源序列和目标序列。


**参数**

    - **src** (Tensor) - Transformer 编码器的输入。它的形状应该是 ``[batch_size, source_length, d_model]``。数据类型为 float32 或是 float64。
    - **tgt** (Tensor) - Transformer 解码器的输入。它的形状应该是 ``[batch_size, target_length, d_model]]``。数据类型为 float32 或是 float64。
    - **src_mask** (Tensor，可选) - 在编码器的多头注意力机制(Multi-head Attention)中，用于避免注意到序列中无关的位置的表示的 Tensor。它的形状应该是，或者能被广播到 ``[batch_size, nhead, source_length, source_length]``。当 ``src_mask`` 的数据类型是 ``bool`` 时，无关的位置所对应的值应该为 ``False`` 并且其余为 ``True``。当 ``src_mask`` 的数据类型为 ``int`` 时，无关的位置所对应的值应该为 0 并且其余为 1。当 ``src_mask`` 的数据类型为 ``float`` 时，无关的位置所对应的值应该为 ``-INF`` 并且其余为 0。当输入中不包含无关项的时候，当前值可以为 ``None``，表示不做 mask 操作。默认值为 ``None``。
    - **tgt_mask** (Tensor，可选) - 在解码器的自注意力机制(Self Attention)中，用于避免注意到序列中无关的位置的表示的 Tensor。它的形状应该是，或者能被广播到 ``[batch_size, nhead, target_length, target_length]``。当 ``src_mask`` 的数据类型是 ``bool`` 时，无关的位置所对应的值应该为 ``False`` 并且其余为 ``True``。当 ``src_mask`` 的数据类型为 ``int`` 时，无关的位置所对应的值应该为 0 并且其余为 1。当 ``src_mask`` 的数据类型为 ``float`` 时，无关的位置所对应的值应该为 ``-INF`` 并且其余为 0。当输入中不包含无关项的时候，当前值可以为 ``None``，表示不做 mask 操作。默认值为 ``None``。
    - **memory_mask** (Tensor，可选) - 在解码器的交叉注意力机制(Cross Attention)中，用于避免注意到序列中无关的位置的表示的 Tensor，通常情况下指的是 padding 的部分。它的形状应该是，或者能被广播到 ``[batch_size, nhead, target_length, source_length]``。当 ``src_mask`` 的数据类型是 ``bool`` 时，无关的位置所对应的值应该为 ``False`` 并且其余为 ``True``。当 ``src_mask`` 的数据类型为 ``int`` 时，无关的位置所对应的值应该为 0 并且其余为 1。当 ``src_mask`` 的数据类型为 ``float`` 时，无关的位置所对应的值应该为 ``-INF`` 并且其余为 0。当输入中不包含无关项的时候，当前值可以为 ``None``，表示不做 mask 操作。默认值为 ``None``。


**返回**

Tensor，Transformer 解码器的输出。其形状和数据类型与 ``tgt`` 相同。



generate_square_subsequent_mask(self, length)
'''''''''

生成一个方形的掩码并且生成的掩码确保对于位置 i 的预测只依赖于已知的结果，即位置小于 i 所对应的结果。


**参数**

    - **length** (int|Tensor) - 序列的长度，``length`` 的数据类型为 int32 或者 int64。若为 Tensor，则当前 Tensor 需仅包含一个数值。


**返回**

Tensor，根据输入的 ``length`` 具体的大小生成的形状为 ``[length, length]`` 方形的掩码。


**代码示例**

COPY-FROM: paddle.nn.Transformer.generate_square_subsequent_mask
