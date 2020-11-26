.. _cn_api_paddle_functional_pixel_shuffle:

pixel_shuffle
-------------------------------

.. py:function:: paddle.nn.functional.pixel_shuffle(x, upscale_factor)




该OP将一个形为[N, C, H, W]的Tensor重新排列成形为 [N, C/r**2, H*r, W*r] 的Tensor。这样做有利于实现步长（stride）为1/r的高效sub-pixel（亚像素）卷积。详见Shi等人在2016年发表的论文 `Real Time Single Image and Video Super Resolution Using an Efficient Sub Pixel Convolutional Neural Network <https://arxiv.org/abs/1609.05158v2>`_ 。

.. code-block:: text

    给定一个形为  x.shape = [1, 9, 4, 4]  的4-D张量
    设定：upscale_factor=3
    那么输出张量的形为：[1, 1, 12, 12]

参数：
          - **x** （Tensor）- 维度为 :math:`[N_1, N_2, ..., N_k, D]` 的多维Tensor，其中最后一维D是类别数目。数据类型为float32或float64。
          - **upscale_factor** （int）- 增大空间分辨率的增大因子
          - **data_format** （str）- 输入和输出数据的数据格式。可从“NCHW”、“NHWC”中选择。默认值为“NCHW”。选择“NCHW”时，数据存储顺序为：[批次大小、输入通道、输入高度、输入宽度]。
          - **name** （str，optional）- 默认值为None。通常用户不需要设置此属性。


返回：根据新的维度信息进行重组的张量

抛出异常： ``ValueError``  - 如果upscale_factor的平方不能整除输入的通道维度(C)的大小。


**示例代码**

..  code-block:: python

    import paddle
    import paddle.nn.functional as F
    import numpy as np
    x = np.random.randn(2, 9, 4, 4).astype(np.float32)
    x_var = paddle.to_tensor(x)
    out_var = F.pixel_shuffle(x_var, 3)
    out = out_var.numpy()
    # (2, 1, 12, 12)

