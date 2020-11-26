.. _cn_api_nn_functional_pixel_shuffle:


pixel_shuffle
-------------------------------

.. py:function:: paddle.nn.functional.pixel_shuffle(x, upscale_factor, data_format="NCHW", name=None)
该算子将一个形为[N, C, H, W]或是[N, H, W, C]的Tensor重新排列成形为 [N, C/r**2, H*r, W*r]或 [N, H*r, W*r, C/r**2]  的Tensor。这样做有利于实现步长（stride）为1/r的高效sub-pixel（亚像素）卷积。详见Shi等人在2016年发表的论文 `Real Time Single Image and Video Super Resolution Using an Efficient Sub Pixel Convolutional Neural Network <https://arxiv.org/abs/1609.05158v2>`_ 。

.. note::
   详细请参考对应的 `Class` 请参考: :ref:`cn_api_nn_PixelShuffle` 。

参数
:::::::::
    - **x** (Tensor): 当前算子的输入, 其是一个形状为 `[N, C, H, W]` 的4-D Tensor。其中 `N` 是batch size, `C` 是通道数, `H` 是输入特征的高度, `W` 是输入特征的宽度。 其数据类型为float32或者float64。
    - **upscale_factor** （int):增大空间分辨率的增大因子
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
    x = paddle.randn((2, 9, 4, 4))
    out = F.pixel_shuffle(x, 3)
    print(out.shape) 
    # (2, 1, 12, 12)
