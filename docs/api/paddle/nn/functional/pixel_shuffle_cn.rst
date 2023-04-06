.. _cn_api_nn_functional_pixel_shuffle:


pixel_shuffle
-------------------------------

.. py:function:: paddle.nn.functional.pixel_shuffle(x, upscale_factor, data_format="NCHW", name=None)

将一个形为 :math:`[N, C, H, W]` 或 :math:`[N, H, W, C]` 的 Tensor 重新排列成形为 :math:`[N, C/r^2, H \times r, W \times r]` 或 :math:`[N, H \times r, W \times r, C/r^2]` 的 Tensor。这样做有利于实现步长（stride）为 1/r 的高效 sub-pixel（亚像素）卷积。详见 Shi 等人在 2016 年发表的论文 `Real Time Single Image and Video Super Resolution Using an Efficient Sub Pixel Convolutional Neural Network <https://arxiv.org/abs/1609.05158v2>`_ 。

.. note::
   详细请参考对应的 `Class` 请参考：:ref:`cn_api_nn_PixelShuffle` 。

参数
:::::::::
    - **x** (Tensor) - 当前算子的输入，其是一个形状为 `[N, C, H, W]` 的 4-D Tensor。其中 `N` 是 batch size, `C` 是通道数，`H` 是输入特征的高度，`W` 是输入特征的宽度。其数据类型为 float16，float32，float64, uint16。
    - **upscale_factor** （int) - 增大空间分辨率的增大因子
    - **data_format** (str，可选) - 数据格式，可选："NCHW"或"NHWC"，默认："NCHW"
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
``Tensor``，输出 Tensor，其数据类型与输入相同。

代码示例
:::::::::

COPY-FROM: paddle.nn.functional.pixel_shuffle
