.. _cn_api_paddle_nn_TransformerDecoder:

TransformerDecoder
-------------------------------

.. py:class:: paddle.nn.TransformerDecoder(decoder_layer, num_layers, norm=None)



**Transformer 解码器**

Transformer 解码器由多个 Transformer 解码器层（``TransformerDecoderLayer``）叠加组成的。


参数
::::::::::::

    - **decoder_layer** (Layer) - ``TransformerDecoderLayer`` 的一个实例，作为 Transformer 解码器的第一层，其他层将根据它的配置进行构建。
    - **num_layers** (int) - ``TransformerDecoderLayer`` 层的叠加数量。
    - **norm** (LayerNorm，可选) - 层标准化（Layer Normalization）。如果提供该参数，将对解码器的最后一层的输出进行层标准化。


代码示例
::::::::::::

COPY-FROM: paddle.nn.TransformerDecoder
