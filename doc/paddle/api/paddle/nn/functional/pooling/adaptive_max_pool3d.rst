.. _cn_api_nn_functional_adaptive_max_pool3d:

adaptive_max_pool3d
-------------------------------

.. py:function:: paddle.nn.functional.adaptive_max_pool3d(x, output_size, return_mask=False, name=None)
该算子根据输入 `x` , `output_size` 等参数对一个输入Tensor计算3D的自适应最大值池化。输入和输出都是5-D Tensor，
默认是以 `NCDHW` 格式表示的，其中 `N` 是 batch size, `C` 是通道数, `D` , `H` , `W` 是输入特征的深度，高度，宽度.

.. note::
   详细请参考对应的 `Class` 请参考: :ref:`cn_api_nn_AdaptiveMaxPool3D` 。


参数
:::::::::
    - **x** (Tensor): 当前算子的输入, 其是一个形状为 `[N, C, D, H, W]` 的5-D Tensor。其中 `N` 是batch size, `C` 是通道数, `D` , `H` , `W` 是输入特征的深度，高度，宽度。 其数据类型为float32或者float64。
    - **output_size** (int|list|tuple): 算子输出特征图的长度，其数据类型为int或list，tuple。
    - **return_mask** (bool, 可选): 如果设置为True，则会与输出一起返回最大值的索引，默认为False。
    - **name** (str，可选): 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
:::::::::
``Tensor``, 输入 `x` 经过自适应池化计算得到的目标5-D Tensor，其数据类型与输入相同。


代码示例
:::::::::

.. code-block:: python

        # adaptive max pool3d
        # suppose input data in the shape of [N, C, D, H, W], `output_size` is [l, m, n]
        # output shape is [N, C, l, m, n], adaptive pool divide D, H and W dimensions
        # of input data into m*n grids averagely and performs poolings in each
        # grid to get output.
        # adaptive max pool performs calculations as follow:
        #
        #     for i in range(l):
        #         for j in range(m):
        #             for k in range(n):
        #                 dstart = floor(i * D / l)
        #                 dend = ceil((i + 1) * D / l)
        #                 hstart = floor(i * H / m)
        #                 hend = ceil((i + 1) * H / m)
        #                 wstart = floor(i * W / n)
        #                 wend = ceil((i + 1) * W / n)
        #             output[:, :, i, j, k] = max(input[:, :, dstart: dend, hstart: hend, wstart: wend])
        #

        import paddle
        x = paddle.rand((2, 3, 8, 32, 32))
        # x.shape is [2, 3, 8, 32, 32]
        out = paddle.nn.functional.adaptive_max_pool3d(
                        x = x,
                        output_size=[3, 3, 3])
        print(out.shape)
        # out.shape is [2, 3, 3, 3, 3]