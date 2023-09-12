.. _cn_api_paddle_nn_PixelUnshuffle:

PixelUnshuffle
-------------------------------

.. py:function:: paddle.nn.PixelUnshuffle(downscale_factor, data_format="NCHW", name=None)
将一个形为 :math:`[N, C, H, W]` 或是 :math:`[N, H, W, C]` 的 Tensor 重新排列成形为 :math:`[N, r^2C, H/r, W/r]` 或 :math:`[N, H/r, W/r, r^2C]` 的 Tensor，这里 :math:`r` 是减小空间分辨率的减小因子。这个算子是 PixelShuffle 算子（请参考：:ref:`cn_api_paddle_nn_PixelShuffle`）的逆算子。详见施闻哲等人在 2016 年发表的论文 `Real Time Single Image and Video Super Resolution Using an Efficient Sub Pixel Convolutional Neural Network <https://arxiv.org/abs/1609.05158v2>`_ 。

.. code-block:: text

    给定一个形为 x.shape = [1, 1, 12, 12] 的 4-D Tensor
    设定 downscale_factor = 3
    那么输出 Tensor 的形为 [1, 9, 4, 4]

参数
:::::::::
    - **downscale_factor** (int) – 减小空间分辨率的减小因子。
    - **data_format** (str，可选) – 数据格式，可选 NCHW 或 NHWC，默认为 NCHW，即（批大小，通道数，高度，宽度）的格式。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
:::::::::
    - **x** (Tensor) – 形状为 :math:`[N, C, H, W]` 或 :math:`[N, C, H, W]` 的 4-D Tensor。
    - **out** (Tensor) – 形状为 :math:`[N, r^2C, H/r, W/r]` 或 :math:`[N, H/r, W/r, r^2C]` 的 4-D Tensor，这里 :math:`r` 就是 :attr:`downscale_factor`。

代码示例
:::::::::

COPY-FROM: paddle.nn.PixelUnshuffle
