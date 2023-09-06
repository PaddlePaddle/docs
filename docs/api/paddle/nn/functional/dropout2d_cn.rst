.. _cn_api_nn_functional_dropout2d:

dropout2d
-------------------------------

.. py:function:: paddle.nn.functional.dropout2d(x, p=0.5, training=True, data_format='NCHW', name=None)

根据丢弃概率 `p`，在训练过程中随机将某些通道特征图置 0 (对一个形状为 `NCHW` 的 4 维 Tensor，通道特征图指的是其中的形状为 `HW` 的 2 维特征图)。

基于 ``paddle.nn.functional.dropout`` 实现，如您想了解更多，请参见 :ref:`cn_api_nn_functional_dropout` 。

参数
:::::::::
 - **x** (Tensor) - 形状为 [N, C, H, W] 或 [N, H, W, C] 的 4D `Tensor`。数据类型为 float16、float32 或 float64。
 - **p** (float，可选) - 将输入通道置 0 的概率，即丢弃概率。默认值为 0.5。
 - **training** (bool，可选) - 标记是否为训练阶段。默认值为 True。
 - **data_format** (str，可选) - 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是 `NCHW` 和 `NHWC`。其中 `N` 是批尺寸，`C` 是通道数，`H` 是特征高度，`W` 是特征宽度。默认值为 `NCHW` 。
 - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
经过 dropout2d 之后的结果，与输入 x 形状相同的 `Tensor` 。

代码示例
:::::::::

COPY-FROM: paddle.nn.functional.dropout2d
