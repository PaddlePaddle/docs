.. _cn_api_nn_TransformerDecoder:

TransformerDecoder
-------------------------------

.. py:class:: paddle.nn.TransformerDecoder(decoder_layer, num_layers, norm=None)



**Transformer解码器**

Transformer解码器由多个Transformer解码器层（``TransformerDecoderLayer``）叠加组成的。


参数：
    - **decoder_layer** (Layer) - ``TransformerDecoderLayer`` 的一个实例，作为Transformer解码器的第一层，其他层将根据它的配置进行构建。
    - **num_layers** (int) - ``TransformerDecoderLayer`` 层的叠加数量。
    - **norm** (LayerNorm，可选) - 层标准化（Layer Normalization）。如果提供该参数，将对解码器的最后一层的输出进行层标准化。


**代码示例**：

.. code-block:: python

   import paddle
   from paddle.nn import TransformerDecoderLayer, TransformerDecoder
   
   # decoder input: [batch_size, tgt_len, d_model]
   dec_input = paddle.rand((2, 4, 128))
   # encoder output: [batch_size, src_len, d_model]
   enc_output = paddle.rand((2, 6, 128))
   # self attention mask: [batch_size, n_head, tgt_len, tgt_len]
   self_attn_mask = paddle.rand((2, 2, 4, 4))
   # cross attention mask: [batch_size, n_head, tgt_len, src_len]
   cross_attn_mask = paddle.rand((2, 2, 4, 6))
   decoder_layer = TransformerDecoderLayer(128, 2, 512)
   decoder = TransformerDecoder(decoder_layer, 2)
   output = decoder(dec_input,
                    enc_output,
                    self_attn_mask,
                    cross_attn_mask)  # [2, 4, 128]
   
