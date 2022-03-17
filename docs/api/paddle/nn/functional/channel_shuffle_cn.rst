.. _cn_api_nn_functional_channel_shuffle:


channel_shuffle
-------------------------------

.. py:function:: paddle.nn.functional.channel_shuffle(x, groups, data_format="NCHW", name=None)
该算子将一个形为[N, C, H, W]或是[N, H, W, C]的Tensor按通道分成g组，得到形为[N, g, C/g, H, W]或[N, H, W, g, C/g]的Tensor，然后转置为[N, C/g, g, H, W]或[N, H, W, C/g, g]的形状，最后重塑为原来的形状。这样做可以增加通道间的信息流动，提高特征的重用率。详见张祥雨等人在2017年发表的论文 `ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices <https://arxiv.org/abs/1707.01083>`_ 。

.. note::
   详细请参考对应的 `Class` 请参考: :ref:`cn_api_nn_ChannelShuffle` 。

参数
:::::::::
    - **x** (Tensor): 当前算子的输入, 其是一个形状为 `[N, C, H, W]` 的4-D Tensor。其中 `N` 是batch size, `C` 是通道数, `H` 是输入特征的高度, `W` 是输入特征的宽度。 其数据类型为float32或者float64。
    - **groups** （int):要把通道分成的组数
    - **data_format** (str，可选): 数据格式，可选："NCHW"或"NHWC"，默认:"NCHW"
    - **name** (str，可选): 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
:::::::::
``Tensor``, 输出Tensor, 其数据类型与输入相同。

代码示例
:::::::::

.. code-block:: python
        
    import paddle
    import paddle.nn.functional as F
    x = paddle.arange(0, 0.6, 0.1, 'float32')
    x = paddle.reshape(x, [1, 6, 1, 1])
    # [[[[0.        ]],
    #   [[0.10000000]],
    #   [[0.20000000]],
    #   [[0.30000001]],
    #   [[0.40000001]],
    #   [[0.50000000]]]]
    y = F.channel_shuffle(x, 3)
    # [[[[0.        ]],
    #   [[0.20000000]],
    #   [[0.40000001]],
    #   [[0.10000000]],
    #   [[0.30000001]],
    #   [[0.50000000]]]]
