.. _cn_api_paddle_incubate_nn_functional_fused_bias_dropout_residual_layer_norm:

fused_bias_dropout_residual_layer_norm
-------------------------------

.. py:function:: paddle.incubate.nn.functional.fused_bias_dropout_residual_layer_norm(x, residual, bias=None, ln_scale=None, ln_bias=None, dropout_rate=0.5, ln_epsilon=1e-05, training=True, mode='upscale_in_train', name=None)

融合偏置、Dropout 和残差层归一化操作符。其伪代码如下：

.. code-block:: text

    >>> y = layer_norm(residual + dropout(bias + x))

参数
::::::::::::
    - **x** (Tensor) - 输入张量。其形状为 `[*, embed_dim]`。
    - **residual** (Tensor) - 残差张量。其形状与 x 相同。
    - **bias** (Tensor，可选) - 线性的偏置。其形状为 `[embed_dim]`。默认为 None。
    - **ln_scale** (Tensor，可选) - 层归一化的权重张量。其形状为 `[embed_dim]`。默认为 None。
    - **ln_bias** (Tensor，可选) - 层归一化的偏置张量。其形状为 `[embed_dim]`。默认为 None。
    - **dropout_rate** (float，可选) - 在注意力权重上使用的 Dropout 概率，用于在注意力后的 Dropout 过程中丢弃一些注意力目标。0 表示无 Dropout。默认为 0.5。
    - **ln_epsilon** (float，可选) - 在层归一化的分母中添加的小浮点数，用于避免除以零。默认为 1e-5。
    - **training** (bool，可选) - 表示是否处于训练阶段的标志。默认为 True。
    - **mode** (str，可选) - ['upscale_in_train'(默认) | 'downscale_in_infer']，有两种模式：

                                 1. upscale_in_train(默认)，在训练时上调输出
                                    - 训练：out = input * mask / (1.0 - p)
                                    - 推理：out = input

                                 2. downscale_in_infer，在推理时下调输出
                                    - 训练：out = input * mask
                                    - 推理：out = input * (1.0 - p)
    - **name** (str，可选) - 操作的名称（可选，默认为 None）。更多信息，请参考：ref:`api_guide_Name`。

返回
::::::::::::
    - Tensor，输出张量，数据类型和形状与 `x` 相同。


代码示例
::::::::::::

COPY-FROM: paddle.incubate.nn.functional.fused_bias_dropout_residual_layer_norm
