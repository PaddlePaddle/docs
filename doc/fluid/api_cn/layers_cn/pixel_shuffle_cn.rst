.. _cn_api_fluid_layers_pixel_shuffle:

pixel_shuffle
-------------------------------

.. py:function:: paddle.fluid.layers.pixel_shuffle(x, upscale_factor)

pixel shuffle 层（像素重组层）

该层将一个形为[N, C, H, W]的张量重新排列成形为 [N, C/r**2, H*r, W*r] 的张量。这样做有利于实现步长（stride）为1/r的高效sub-pixel（亚像素）卷积。详见Shi等人在2016年发表的论文 `Real Time Single Image and Video Super Resolution Using an Efficient Sub Pixel Convolutional Neural Network <https://arxiv.org/abs/1609.05158v2>`_ 。

.. code-block:: text

    给定一个形为  x.shape = [1, 9, 4, 4]  的4-D张量
    设定：upscale_factor=3
    那么输出张量的形为：[1, 1, 12, 12]

参数：
          - **x** （Variable）- 输入Tensor变量。
          - **upscale_factor** （int）- 增大空间分辨率的增大因子


返回：根据新的维度信息进行重组的张量

返回类型：  Variable

抛出异常： ``ValueError``  - 如果upscale_factor的平方不能整除输入的通道维(C)大小。


**示例代码**

..  code-block:: python

    import paddle.fluid as fluid
    input = fluid.layers.data(name="input", shape=[9,4,4])
    output = fluid.layers.pixel_shuffle(x=input, upscale_factor=3)





