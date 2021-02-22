.. _cn_api_nn_AdaptiveMaxPool3D:


AdaptiveMaxPool3D
-------------------------------

.. py:function:: paddle.nn.AdaptiveMaxPool3D(output_size, return_mask=False, name=None)
该算子根据输入 `x` , `output_size` 等参数对一个输入Tensor计算3D的自适应最大池化。输入和输出都是5-D Tensor，
默认是以 `NCDHW` 格式表示的，其中 `N` 是 batch size， `C` 是通道数， `D` ， `H` ， `W` 分别是输入特征的深度，高度，宽度.

计算公式如下:

..  math::

    dstart &= floor(i * D_{in} / D_{out})
    
    dend &= ceil((i + 1) * D_{in} / D_{out})
    
    hstart &= floor(j * H_{in} / H_{out})
    
    hend &= ceil((j + 1) * H_{in} / H_{out})
    
    wstart &= floor(k * W_{in} / W_{out})
    
    wend &= ceil((k + 1) * W_{in} / W_{out})
    
    Output(i ,j, k) &= max(Input[dstart:dend, hstart:hend, wstart:wend])

参数
:::::::::
    - **output_size** (int|list|tuple): 算子输出特征图的高宽长大小，其数据类型为int,list或tuple。
    - **return_mask** (bool): 如果设置为True，则会与输出一起返回最大值的索引，默认为False。
    - **name** (str，可选): 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

形状
:::::::::
    - **x** (Tensor): 默认形状为（批大小，通道数，输出特征深度，高度，宽度），即NCDHW格式的5-D Tensor。 其数据类型为float32或者float64。
    - **output** (Tensor): 默认形状为（批大小，通道数，输出特征深度，高度，宽度），即NCDHW格式的5-D Tensor。 其数据类型与输入x相同。

返回
:::::::::
计算AdaptiveMaxPool3D的可调用对象


代码示例
:::::::::

.. code-block:: python

        # adaptive max pool3d
        # suppose input data in shape of [N, C, D, H, W], `output_size` is [l, m, n],
        # output shape is [N, C, l, m, n], adaptive pool divide D, H and W dimensions
        # of input data into l * m * n grids averagely and performs poolings in each
        # grid to get output.
        # adaptive max pool performs calculations as follow:
        #
        #     for i in range(l):
        #         for j in range(m):
        #             for k in range(n):
        #                 dstart = floor(i * D / l)
        #                 dend = ceil((i + 1) * D / l)
        #                 hstart = floor(j * H / m)
        #                 hend = ceil((j + 1) * H / m)
        #                 wstart = floor(k * W / n)
        #                 wend = ceil((k + 1) * W / n)
        #                 output[:, :, i, j, k] =
        #                     max(input[:, :, dstart:dend, hstart: hend, wstart: wend])

        import paddle
        x = paddle.rand((2, 3, 8, 32, 32))
        pool = paddle.nn.AdaptiveMaxPool3D(output_size=4)
        out = pool(x)
        print(out.shape)
        # out shape: [2, 3, 4, 4, 4]
        pool = paddle.nn.AdaptiveMaxPool3D(output_size=3, return_mask=True)
        out, indices = pool(x)
        print(out.shape)
        print(indices.shape)
        # out shape: [2, 3, 3, 3, 3], indices shape: [2, 3, 3, 3, 3]
