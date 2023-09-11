.. _cn_api_paddle_nn_functional_channel_shuffle:


channel_shuffle
-------------------------------

.. py:function:: paddle.nn.functional.channel_shuffle(x, groups, data_format="NCHW", name=None)
将一个形为 [N, C, H, W] 或是 [N, H, W, C] 的 Tensor 按通道分成 g 组，得到形为 [N, g, C/g, H, W] 或 [N, H, W, g, C/g] 的 Tensor，然后转置为 [N, C/g, g, H, W] 或 [N, H, W, C/g, g] 的形状，最后重塑为原来的形状。这样做可以增加通道间的信息流动，提高特征的重用率。详见张祥雨等人在 2017 年发表的论文 `ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices <https://arxiv.org/abs/1707.01083>`_ 。

.. note::
   详细请参考对应的 `Class` 请参考：:ref:`cn_api_paddle_nn_ChannelShuffle`。

参数
:::::::::
    - **x** (Tensor) – 当前算子的输入，其是一个形状为 [N, C, H, W] 的 4-D Tensor。其中 N 是批大小，C 是通道数，H 是输入特征的高度，W 是输入特征的宽度。其数据类型为 float32 或 float64。
    - **groups** (int) – 要把通道分成的组数；
    - **data_format** (str，可选) – 数据格式，可选：NCHW 或 NHWC，默认为 NCHW，即（批大小，通道数，高度，宽度）的格式。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
``Tensor``，调整过通道顺序的 Tensor，其数据类型与输入相同。

代码示例
:::::::::

COPY-FROM: paddle.nn.functional.channel_shuffle
