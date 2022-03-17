.. _cn_api_nn_ChannelShuffle:

ChannelShuffle
-------------------------------

.. py:function:: paddle.nn.ChannelShuffle(groups, data_format="NCHW", name=None)
该算子将一个形为[N, C, H, W]或是[N, H, W, C]的Tensor按通道分成g组，得到形为[N, g, C/g, H, W]或[N, H, W, g, C/g]的Tensor，然后转置为[N, C/g, g, H, W]或[N, H, W, C/g, g]的形状，最后重塑为原来的形状。这样做可以增加通道间的信息流动，提高特征的重用率。详见张祥雨等人在2017年发表的论文 `ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices <https://arxiv.org/abs/1707.01083>`_ 。


参数
:::::::::
    - **groups** (int): 要把通道分成的组数；
    - **data_format** (str，可选): 数据格式，可选："NCHW"或"NHWC"，默认: "NCHW"；
    - **name** (str，可选): 操作的名称(可选，默认值为None)。更多信息请参见 :ref:`api_guide_Name`。

形状
:::::::::
    - **x** (Tensor): 默认形状为 (批大小，通道数，高度，宽度)，即NCHW格式的4-D Tensor。其数据类型为float32, float64；
    - **output** (Tensor): 其形状与数据类型均和输入相同。

返回
:::::::::
计算ChannelShuffle的可调用对象。

代码示例
:::::::::
.. code-block:: python

    import paddle
    import paddle.nn as nn
    x = paddle.arange(0, 0.6, 0.1, 'float32')
    x = paddle.reshape(x, [1, 6, 1, 1])
    # [[[[0.        ]],
    #   [[0.10000000]],
    #   [[0.20000000]],
    #   [[0.30000001]],
    #   [[0.40000001]],
    #   [[0.50000000]]]]
    channel_shuffle = nn.ChannelShuffle(3)
    y = channel_shuffle(x)
    # [[[[0.        ]],
    #   [[0.20000000]],
    #   [[0.40000001]],
    #   [[0.10000000]],
    #   [[0.30000001]],
    #   [[0.50000000]]]]
