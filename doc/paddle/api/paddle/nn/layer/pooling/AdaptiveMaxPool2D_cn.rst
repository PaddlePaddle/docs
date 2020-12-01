.. _cn_api_nn_AdaptiveMaxPool2D:


AdaptiveMaxPool2D
-------------------------------

.. py:class:: paddle.nn.AdaptiveMaxPool2D(output_size, return_mask=False, name=None)
该算子根据输入 `x` , `output_size` 等参数对一个输入Tensor计算2D的自适应平均池化。输入和输出都是4-D Tensor，
默认是以 `NCHW` 格式表示的，其中 `N` 是 batch size, `C` 是通道数, `H` 是输入特征的高度, `W` 是输入特征的宽度.

计算公式如下:

..  math::

    lstart &= floor(i * L_{in} / L_{out})

    lend &= ceil((i + 1) * L_{in} / L_{out})

    Output(i) &= max(Input[lstart:lend])

    hstart &= floor(i * H_{in} / H_{out})
    
    hend &= ceil((i + 1) * H_{in} / H_{out})
    
    wstart &= floor(j * W_{in} / W_{out})
    
    wend &= ceil((j + 1) * W_{in} / W_{out})
    
    Output(i ,j) &= max(Input[hstart:hend, wstart:wend])

参数
:::::::::

    - **output_size** (int|list|tuple): 算子输出特征图的高和宽大小，其数据类型为int,list或tuple。
    - **return_mask** (bool): 如果设置为True，则会与输出一起返回最大值的索引，默认为False。
    - **name** (str，可选): 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

形状
:::::::::

    - **x** (Tensor): 默认形状为（批大小，通道数，输出特征长度，宽度），即NCHW格式的4-D Tensor。 其数据类型为float32或者float64。
    - **output** (Tensor): 默认形状为（批大小，通道数，输出特征长度，宽度），即NCHW格式的4-D Tensor。 其数据类型与输入x相同。

返回
:::::::::

计算AdaptiveMaxPool2D的可调用对象


代码示例
:::::::::

.. code-block:: python
        
        # adaptive max pool2d
        # suppose input data in shape of [N, C, H, W], `output_size` is [m, n],
        # output shape is [N, C, m, n], adaptive pool divide H and W dimensions
        # of input data into m * n grids averagely and performs poolings in each
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
        x = paddle.randn((2, 3, 32, 32))
        adaptive_max_pool = paddle.nn.AdaptiveMaxPool2D(output_size=3, return_mask=True)
        pool_out, indices = adaptive_max_pool(x = x)
        print(pool_out.shape) # [2, 3, 3, 3]
        print(indices.shape) # [2, 3, 3, 3]
