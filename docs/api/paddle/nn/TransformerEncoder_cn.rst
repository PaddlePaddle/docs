.. _cn_api_nn_TransformerEncoder:

TransformerEncoder
-------------------------------

.. py:class:: paddle.nn.TransformerEncoder(encoder_layer, num_layers, norm=None)



**Transformer 编码器**

Transformer 编码器由多个 Transformer 编码器层（``TransformerEncoderLayer``）叠加组成的。


参数
::::::::::::

    - **encoder_layer** (Layer) - ``TransformerEncoderLayer`` 的一个实例，作为 Transformer 编码器的第一层，其他层将根据它的配置进行构建。
    - **num_layers** (int) - ``TransformerEncoderLayer`` 层的叠加数量。
    - **norm** (LayerNorm，可选) - 层标准化（Layer Normalization）。如果提供该参数，将对编码器的最后一层的输出进行层标准化。


代码示例
::::::::::::

COPY-FROM: paddle.nn.TransformerEncoder
