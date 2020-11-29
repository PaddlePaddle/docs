.. _cn_api_nn_Transformer:

Transformer
-------------------------------

.. py:class:: paddle.nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', attn_dropout=None, act_dropout=None, normalize_before=False, weight_attr=None, bias_attr=None, custom_encoder=None, custom_decoder=None)



**Transformer模型**

Transformer模型由一个 ``TransformerEncoder`` 实例和一个 ``TransformerDecoder`` 实例组成，不包含embedding层和输出层。

细节可参考论文 `Attention is all you need <https://arxiv.org/pdf/1706.03762.pdf>`_ 。

用户可以使用相应的参数配置模型结构。请注意 ``normalize_before`` 的用法与某些类似Transformer的模型例如BERT和GPT2的用法不同，它表示在哪里（多头注意力机制或前馈神经网络的输入还是输出）进行层标准化（Layer Normalization）。该模型默认的结构是对每个子层的output进行层归一化，并在最后一个编码器/解码器的输出上进行另一个层归一化操作。


参数：
    - **d_model** (int，可选) - 编码器和解码器的输入输出的维度。默认值：512。
    - **nhead** (int，可选) - 多头注意力机制的Head数量。默认值：8。
    - **num_encoder_layers** (int，可选) - 编码器中 ``TransformerEncoderLayer`` 的层数。默认值：6。
    - **num_decoder_layers** (int，可选) - 解码器中 ``TransformerDecoderLayer`` 的层数。默认值：6。
    - **dim_feedforward** (int，可选) - 前馈神经网络中隐藏层的大小。默认值：2048。
    - **dropout** (float，可选) - 对编码器和解码器中每个子层的输出进行处理的dropout值。默认值：0.1。
    - **activation** (str，可选) - 前馈神经网络的激活函数。默认值：``relu``。
    - **attn_dropout** (float，可选) - 多头自注意力机制中对注意力目标的随机失活率。如果为 ``None`` 则 ``attn_dropout = dropout``。默认值：``None``。
    - **act_dropout** (float，可选) - 前馈神经网络的激活函数后的dropout。如果为 ``None`` 则 ``act_dropout = dropout``。默认值：``None``。
    - **normalize_before** (bool, 可选) - 设置对编码器解码器的每个子层的输入输出的处理。如果为 ``True``，则对每个子层的输入进行层标准化（Layer Normalization），对每个子层的输出进行dropout和残差连接（residual connection）。否则（即为 ``False``），则对每个子层的输入不进行处理，只对每个子层的输出进行dropout、残差连接（residual connection）和层标准化（Layer Normalization）。默认值：``False``。
    - **weight_attr** (ParamAttr|tuple，可选) - 指定权重参数属性的对象。如果是 ``tuple``，则只支持 ``tuple`` 长度为1、2或3的情况。如果 ``tuple`` 长度为3，多头自注意力机制的权重参数属性使用 ``weight_attr[0]``，解码器的编码-解码交叉注意力机制的权重参数属性使用 ``weight_attr[1]``，前馈神经网络的权重参数属性使用 ``weight_attr[2]``；如果 ``tuple`` 的长度是2，多头自注意力机制和解码器的编码-解码交叉注意力机制的权重参数属性使用 ``weight_attr[0]``，前馈神经网络的权重参数属性使用 ``weight_attr[1]``；如果 ``tuple`` 的长度是1，多头自注意力机制、解码器的编码-解码交叉注意力机制和前馈神经网络的权重参数属性都使用 ``weight_attr[0]``。如果该参数值是 ``ParamAttr``，则多头自注意力机制、解码器的编码-解码交叉注意力机制和前馈神经网络的权重参数属性都使用 ``ParamAttr``。默认值：``None``，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **bias_attr** （ParamAttr|tuple|bool，可选）- 指定偏置参数属性的对象。如果是 ``tuple``，则只支持 ``tuple`` 长度为1、2或3的情况。如果 ``tuple`` 长度为3，多头自注意力机制的偏置参数属性使用 ``bias_attr[0]``，解码器的编码-解码交叉注意力机制的偏置参数属性使用 ``bias_attr[1]``，前馈神经网络的偏置参数属性使用 ``bias_attr[2]``；如果 ``tuple`` 的长度是2，多头自注意力机制和解码器的编码-解码交叉注意力机制的偏置参数属性使用 ``bias_attr[0]``，前馈神经网络的偏置参数属性使用 ``bias_attr[1]``；如果 ``tuple`` 的长度是1，多头自注意力机制、解码器的编码-解码交叉注意力机制和前馈神经网络的偏置参数属性都使用 ``bias_attr[0]``。如果该参数值是 ``ParamAttr``，则多头自注意力机制、解码器的编码-解码交叉注意力机制和前馈神经网络的偏置参数属性都使用 ``ParamAttr``。如果该参数为 ``bool`` 类型，只支持为 ``False``，表示没有偏置参数。默认值：``None``，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **custom_encoder** (Layer，可选) - 若提供该参数，则将 ``custom_encoder`` 作为编码器。默认值：``None``。
    - **custom_decoder** (Layer，可选) - 若提供该参数，则将 ``custom_decoder`` 作为解码器。默认值：``None``。


**代码示例**：

.. code-block:: python

   import paddle
   from paddle.nn import Transformer
   
   # src: [batch_size, tgt_len, d_model]
   enc_input = paddle.rand((2, 4, 128))
   # tgt: [batch_size, src_len, d_model]
   dec_input = paddle.rand((2, 6, 128))
   # src_mask: [batch_size, n_head, src_len, src_len]
   enc_self_attn_mask = paddle.rand((2, 2, 4, 4))
   # tgt_mask: [batch_size, n_head, tgt_len, tgt_len]
   dec_self_attn_mask = paddle.rand((2, 2, 6, 6))
   # memory_mask: [batch_size, n_head, tgt_len, src_len]
   cross_attn_mask = paddle.rand((2, 2, 6, 4))
   transformer = Transformer(128, 2, 4, 4, 512)
   output = transformer(enc_input,
                        dec_input,
                        enc_self_attn_mask,
                        dec_self_attn_mask,
                        cross_attn_mask)  # [2, 6, 128]
   
