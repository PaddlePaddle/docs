.. _cn_api_paddle_nn_functional_pixel_unshuffle:

pixel_unshuffle
-------------------------------

.. py:function:: paddle.nn.functional.pixel_unshuffle(x, downscale_factor, data_format="NCHW", name=None)
将一个形为 :math:`[N, C, H, W]` 或 :math:`[N, H, W, C]` 的 Tensor 重新排列成形为 :math:`[N, r^2C, H/r, W/r]` 或 :math:`[N, H/r, W/r, r^2C]` 的 Tensor，这里 :math:`r` 是减小空间分辨率的减小因子。这个算子是 pixel_shuffle 算子（请参考：:ref:`cn_api_paddle_nn_functional_pixel_shuffle`）的逆算子。详见施闻哲等人在 2016 年发表的论文 `Real Time Single Image and Video Super Resolution Using an Efficient Sub Pixel Convolutional Neural Network <https://arxiv.org/abs/1609.05158v2>`_ 。

.. note::
   详细请参考对应的 `Class` 请参考：:ref:`cn_api_paddle_nn_PixelUnshuffle` 。

参数
:::::::::
    - **x** (Tensor) – 当前算子的输入，其是一个形状为 :math:`[N, C, H, W]` 或 :math:`[N, H, W, C]` 的 4-D Tensor。其中 :math:`N` 是批大小，:math:`C` 是通道数，:math:`H` 是输入特征的高度，:math:`W` 是输入特征的宽度。其数据类型为 float32 或 float64。
    - **downscale_factor** (int) – 减小空间分辨率的减小因子。
    - **data_format** (str，可选) – 数据格式，可选 NCHW 或 NHWC，默认为 NCHW，即（批大小，通道数，高度，宽度）的格式。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
``Tensor``，重新排列过的 Tensor，其数据类型与输入相同。

代码示例
:::::::::

COPY-FROM: paddle.nn.functional.pixel_unshuffle
