.. _cn_api_nn_functional_adaptive_max_pool2d:

adaptive_max_pool2d
-------------------------------

.. py:function:: paddle.nn.functional.adaptive_max_pool2d(x, output_size, return_mask=False, name=None)
该算子根据输入 `x` , `output_size` 等参数对一个输入Tensor计算2D的自适应最大值池化。输入和输出都是4-D Tensor，
默认是以 `NCHW` 格式表示的，其中 `N` 是 batch size, `C` 是通道数, `H` 是输入特征的高度， `W` 是输入特征的宽度。

.. note::
   详细请参考对应的 `Class` 请参考: :ref:`cn_api_nn_AdaptiveMaxPool2D` 。


参数
:::::::::
    - **x** (Tensor): 当前算子的输入, 其是一个形状为 `[N, C, H, W]` 的4-D Tensor。其中 `N` 是batch size, `C` 是通道数, `H` 是输入特征的高度, `W` 是输入特征的宽度。 其数据类型为float32或者float64。
    - **output_size** (int|list|tuple): 算子输出特征图的长度，其数据类型为int或list，tuple。
    - **return_mask** (bool, 可选): 如果设置为True，则会与输出一起返回最大值的索引，默认为False。
    - **name** (str，可选): 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
:::::::::
``Tensor``, 输入 `x` 经过自适应池化计算得到的目标4-D Tensor，其数据类型与输入相同。

代码示例
:::::::::

.. code-block:: python
        
        # max adaptive pool2d
        # suppose input data in the shape of [N, C, H, W], `output_size` is [m, n]
        # output shape is [N, C, m, n], adaptive pool divide H and W dimensions
        # of input data into m*n grids averagely and performs poolings in each
        # grid to get output.
        # adaptive max pool performs calculations as follow:
        #
        #     for i in range(m):
        #         for j in range(n):
        #             hstart = floor(i * H / m)
        #             hend = ceil((i + 1) * H / m)
        #             wstart = floor(i * W / n)
        #             wend = ceil((i + 1) * W / n)
        #             output[:, :, i, j] = max(input[:, :, hstart: hend, wstart: wend])
        #
        
        import paddle
        x = paddle.rand((2, 3, 32, 32))
        # x.shape is [2, 3, 32, 32]
        out = paddle.nn.functional.adaptive_max_pool2d(
                        x = x,
                        output_size=[3, 3])
        print(out.shape)
        # out.shape is [2, 3, 3, 3]