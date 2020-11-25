.. _cn_api_nn_TransformerEncoder:

TransformerEncoder
-------------------------------

.. py:class:: paddle.nn.TransformerEncoder(encoder_layer, num_layers, norm=None)



**Transformer编码器**

Transformer编码器由多个Transformer编码器层（``TransformerEncoderLayer``）叠加组成的。


参数：
    - **encoder_layer** (Layer) - ``TransformerEncoderLayer`` 的一个实例，作为Transformer编码器的第一层，其他层将根据它的配置进行构建。
    - **num_layers** (int) - ``TransformerEncoderLayer`` 层的叠加数量。
    - **norm** (LayerNorm，可选) - 层标准化（Layer Normalization）。如果提供该参数，将对编码器的最后一层的输出进行层标准化。


**代码示例**：

.. code-block:: python

   import paddle
   from paddle.nn import TransformerEncoderLayer, TransformerEncoder
   
   # encoder input: [batch_size, src_len, d_model]
   enc_input = paddle.rand((2, 4, 128))
   # self attention mask: [batch_size, n_head, src_len, src_len]
   attn_mask = paddle.rand((2, 2, 4, 4))
   encoder_layer = TransformerEncoderLayer(128, 2, 512)
   encoder = TransformerEncoder(encoder_layer, 2)
   enc_output = encoder(enc_input, attn_mask)  # [2, 4, 128]
   
