.. _cn_api_paddle_nn_ChannelShuffle:

ChannelShuffle
-------------------------------

.. py:function:: paddle.nn.ChannelShuffle(groups, data_format="NCHW", name=None)
将一个形为 [N, C, H, W] 或是 [N, H, W, C] 的 Tensor 按通道分成 g 组，得到形为 [N, g, C/g, H, W] 或 [N, H, W, g, C/g] 的 Tensor，然后转置为 [N, C/g, g, H, W] 或 [N, H, W, C/g, g] 的形状，最后重塑为原来的形状。这样做可以增加通道间的信息流动，提高特征的重用率。详见张祥雨等人在 2017 年发表的论文 `ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices <https://arxiv.org/abs/1707.01083>`_ 。


参数
:::::::::
    - **groups** (int) – 要把通道分成的组数。
    - **data_format** (str，可选) – 数据格式，可选：NCHW 或 NHWC，默认为 NCHW，即（批大小，通道数，高度，宽度）的格式。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
:::::::::
    - **x** (Tensor) – 默认形状为（批大小，通道数，高度，宽度），即 NCHW 格式的 4-D Tensor。其数据类型为 float32 或 float64。
    - **out** (Tensor) – 其形状与数据类型均和输入相同。

返回
:::::::::
计算 ChannelShuffle 的可调用对象。

代码示例
:::::::::

COPY-FROM: paddle.nn.ChannelShuffle
