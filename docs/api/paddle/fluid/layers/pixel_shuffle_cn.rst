.. _cn_api_fluid_layers_pixel_shuffle:

pixel_shuffle
-------------------------------

.. py:function:: paddle.fluid.layers.pixel_shuffle(x, upscale_factor)




该 OP 将一个形为[N, C, H, W]的 Tensor 重新排列成形为 [N, C/r**2, H*r, W*r] 的 Tensor。这样做有利于实现步长（stride）为 1/r 的高效 sub-pixel（亚像素）卷积。详见 Shi 等人在 2016 年发表的论文 `Real Time Single Image and Video Super Resolution Using an Efficient Sub Pixel Convolutional Neural Network <https://arxiv.org/abs/1609.05158v2>`_ 。

.. code-block:: text

    给定一个形为  x.shape = [1, 9, 4, 4]  的 4-DTensor
    设定：upscale_factor=3
    那么输出 Tensor 的形为：[1, 1, 12, 12]

参数
::::::::::::

          - **x** （Variable）- 维度为 :math:`[N_1, N_2, ..., N_k, D]` 的多维 Tensor，其中最后一维 D 是类别数目。数据类型为 float32 或 float64。
          - **upscale_factor** （int）- 增大空间分辨率的增大因子


返回
::::::::::::
根据新的维度信息进行重组的 Tensor

返回类型
::::::::::::
  Variable

抛出异常
::::::::::::
 ``ValueError``  - 如果 upscale_factor 的平方不能整除输入的通道维度(C)的大小。


代码示例
::::::::::::

..  code-block:: python

    import paddle.fluid as fluid
    input = fluid.layers.data(name="input", shape=[9,4,4])
    output = fluid.layers.pixel_shuffle(x=input, upscale_factor=3)
