.. _cn_api_nn_PixelShuffle:

PixelShuffle
-------------------------------

.. py:function:: paddle.nn.PixelShuffle(upscale_factor, data_format="NCHW", name=None)
该算子将一个形为[N, C, H, W]或是[N, H, W, C]的Tensor重新排列成形为 [N, C/r**2, H*r, W*r]或 [N, H*r, W*r, C/r**2]  的Tensor。这样做有利于实现步长（stride）为1/r的高效sub-pixel（亚像素）卷积。详见Shi等人在2016年发表的论文 `Real Time Single Image and Video Super Resolution Using an Efficient Sub Pixel Convolutional Neural Network <https://arxiv.org/abs/1609.05158v2>`_ 。

.. code-block:: text

    给定一个形为  x.shape = [1, 9, 4, 4]  的4-D张量
    设定：upscale_factor=3
    那么输出张量的形为：[1, 1, 12, 12]

参数
:::::::::
    - **upscale_factor** （int):增大空间分辨率的增大因子
    - **data_format** (str，可选): 数据格式，可选："NCHW"或"NHWC"，默认:"NCHW"
    - **name** (str，可选): 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

形状
:::::::::
    - **x** (Tensor): 默认形状为（批大小，通道数，高度，宽度），即NCHW格式的4-D Tensor或NHWC格式的4-DTensor。 其数据类型为float32, float64.
    - **output** (Tensor): 默认形状为（批大小，通道数，输出特征高度，输出特征宽度），即NCHW格式或NHWC的4-D Tensor。 其数据类型与输入相同。

返回
:::::::::
计算PixelShuffle的可调用对象

代码示例
:::::::::
.. code-block:: python

    import paddle
    import paddle.nn as nn

    x = paddle.rand((2, 9, 4, 4))
    pixel_shuffle = nn.PixelShuffle(3)
    out_var = pixel_shuffle(x_var)
    out = out_var.numpy()
    print(out.shape)
    # (2, 1, 12, 12)
