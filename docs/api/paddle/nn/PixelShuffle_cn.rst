.. _cn_api_paddle_nn_PixelShuffle:

PixelShuffle
-------------------------------

.. py:function:: paddle.nn.PixelShuffle(upscale_factor, data_format="NCHW", name=None)

将一个形为 :math:`[N, C, H, W]` 或是 :math:`[N, H, W, C]` 的 Tensor 重新排列成形为 :math:`[N, C/r^2, H \times r, W \times r]` 或 :math:`[N, H \times r, W \times r, C/r^2]` 的 Tensor。这样做有利于实现步长（stride）为 :math:`1/r` 的高效 sub-pixel（亚像素）卷积。详见 Shi 等人在 2016 年发表的论文 `Real Time Single Image and Video Super Resolution Using an Efficient Sub Pixel Convolutional Neural Network <https://arxiv.org/abs/1609.05158v2>`_ 。

.. code-block:: text

    给定一个形为  x.shape = [1, 9, 4, 4]  的 4-D Tensor
    设定：upscale_factor=3
    那么输出 Tensor 的形为：[1, 1, 12, 12]

参数
:::::::::
    - **upscale_factor** （int) - 增大空间分辨率的增大因子
    - **data_format** (str，可选) - 数据格式，可选："NCHW"或"NHWC"，默认："NCHW"
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
:::::::::
    - **x** (Tensor)：默认形状为（批大小，通道数，高度，宽度），即 NCHW 格式的 4-D Tensor 或 NHWC 格式的 4-DTensor。其数据类型为 float32, float64。
    - **output** (Tensor)：默认形状为（批大小，通道数，输出特征高度，输出特征宽度），即 NCHW 格式或 NHWC 的 4-D Tensor。其数据类型与输入相同。

返回
:::::::::
计算 PixelShuffle 的可调用对象

代码示例
:::::::::

COPY-FROM: paddle.nn.PixelShuffle
