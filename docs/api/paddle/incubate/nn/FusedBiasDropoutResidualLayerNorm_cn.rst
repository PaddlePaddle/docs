.. _cn_api_paddle_incubate_nn_FusedBiasDropoutResidualLayerNorm:

FusedBiasDropoutResidualLayerNorm
-------------------------------

.. py:class:: paddle.incubate.nn.FusedBiasDropoutResidualLayerNorm(embed_dim, dropout_rate=0.5, weight_attr=None, bias_attr=None, epsilon=1e-05, name=None)

应用 fused_bias_dropout_residual_layer_norm 操作符，包含融合偏置、Dropout 和残差层归一化操作。

参数
::::::::::::
    - **embed_dim** (int) - 输入和输出中预期的特征大小。
    - **dropout_rate** (float，可选) - 在注意力权重上使用的 Dropout 概率，用于在注意力后的 Dropout 过程中丢弃一些注意力目标。0 表示无 Dropout。默认为 0.5。
    - **bias_attr** (ParamAttr|bool，可选) - 指定偏置参数的属性。默认为 None，意味着使用默认的偏置参数属性。如果设置为 False，则该层不会有可训练的偏置参数。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
    - **epsilon** (float，可选) - 添加到方差中的小值，以防止除零。默认为 1e-05。

代码示例
::::::::::::

COPY-FROM: paddle.incubate.nn.FusedBiasDropoutResidualLayerNorm

forward(x, residual)
::::::::::::
应用 fused_bias_dropout_residual_layer_norm 操作符，包含融合偏置、Dropout 和残差层归一化操作。

参数
::::::::::::
    - **x** (Tensor) - 输入张量。它是一个形状为 `[batch_size, seq_len, embed_dim]` 的张量。数据类型应为 float32 或 float64 。
    - **residual** (Tensor，可选) - 残差张量。它是一个形状为 `[batch_size, value_length, vdim]` 的张量。数据类型应为 float32 或 float64。

返回
::::::::::::
Tensor|tuple：与 `x` 具有相同数据类型和形状的张量

extra_repr()
::::::::::::
当前层的额外表示，您可以自定义实现自己的层。
