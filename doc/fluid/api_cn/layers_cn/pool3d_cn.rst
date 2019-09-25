.. _cn_api_fluid_layers_pool3d:

pool3d
-------------------------------

.. py:function:: paddle.fluid.layers.pool3d(input, pool_size=-1, pool_type='max', pool_stride=1, pool_padding=0, global_pooling=False, use_cudnn=True, ceil_mode=False, name=None, exclusive=True, data_format="NCDHW")

该OP使用上述输入参数的池化配置，为三维空间池化操作，根据 ``input`` ，池化类型 ``pool_type`` ，池化核大小 ``pool_size`` ，步长 ``pool_stride`` 和填充 ``pool_padding`` 参数计算输出。

输入 ``input`` 和输出（Out）采用NCDHW或NDHWC格式，其中N是批大小，C是通道数，D，H和W分别是特征的深度，高度和宽度。

参数 ``ksize`` ，``strides`` 含有三个整型元素。 分别代表深度，高度和宽度上的参数。

输入 ``input`` 和输出（Out）的形状可能不同。


例如：

输入：
   ``X`` 的形状： :math:`(N, C, D_{in}, H_{in}, W_{in})`

输出：
    ``out`` 的形状： :math:`(N, C, D_{out}, H_{out}, W_{out})`

当 ``ceil_mode`` = false时，

.. math::

    D_{out} &= \frac{(D_{in} - ksize[0] + pad\_depth\_front + pad\_depth\_back)}{strides[0]} + 1\\
    H_{out} &= \frac{(H_{in} - ksize[1] + pad\_height\_top + pad\_height\_bottom)}{strides[2]} + 1\\
    W_{out} &= \frac{(W_{in} - ksize[2] + pad\_width\_left + pad\_width\_right)}{strides[2]} + 1

当 ``ceil_mode`` = true时，

.. math::

    D_{out} &= \frac{(D_{in} - ksize[0] + pad\_depth\_front + pad\_depth\_back + strides[0] -1)}{strides[0]} + 1\\
    H_{out} &= \frac{(H_{in} - ksize[1] + pad\_height\_top + pad\_height\_bottom + strides[1] -1)}{strides[1]} + 1\\
    W_{out} &= \frac{(W_{in} - ksize[2] + pad\_width\_left + pad\_width\_right + strides[2] -1)}{strides[2]} + 1

当 ``exclusive`` = false时，

.. math::
    dstart &= i * strides[0] - pad\_depth\_front \\
    dend &= dstart + ksize[0] \\
    hstart &= j * strides[1] - pad\_height\_top \\
    hend &= hstart + ksize[1] \\
    wstart &= k * strides[2] - pad\_width\_left \\
    wend &= wstart + ksize[2] \\
    Output(i ,j, k) &= \frac{sum(Input[dstart:dend, hstart:hend, wstart:wend])}{ksize[0] * ksize[1] * ksize[2]}

如果 ``exclusive`` = true:

.. math::
    dstart &= max(0, i * strides[0] - pad\_depth\_front) \\
    dend &= min(D, dstart + ksize[0]) \\
    hstart &= max(0, j * strides[1] - pad\_height\_top) \\
    hend &= min(H, hstart + ksize[1]) \\
    wstart &= max(0, k * strides[2] - pad\_width\_left) \\
    wend & = min(W, wstart + ksize[2]) \\
    Output(i ,j, k) & = \frac{sum(Input[dstart:dend, hstart:hend, wstart:wend])}{(dend - dstart) * (hend - hstart) * (wend - wstart)}

如果 ``pool_padding`` = "SAME":

.. math::
    D_{out} = \frac{(D_{in} + strides[0] - 1)}{strides[0]}

.. math::
    H_{out} = \frac{(H_{in} + strides[1] - 1)}{strides[1]}

.. math::
    W_{out} = \frac{(W_{in} + strides[2] - 1)}{strides[2]}

如果 ``pool_padding`` = "VALID":

.. math::
    D_{out} = \frac{(D_{in} - ksize[0])}{strides[0]} + 1

.. math::
    H_{out} = \frac{(H_{in} - ksize[1])}{strides[1]} + 1

.. math::
    W_{out} = \frac{(W_{in} - ksize[2])}{strides[2]} + 1


参数：
    - **input** (Vairable) - 池化运算的输入张量, 形状为 :math:`[N, C, D, H, W]` 或  :math:`[N, D, H, W, C]` 的5-D Tensor，N是批尺寸，C是通道数，D是特征深度，H是特征高度，W是特征宽度，数据类型为float32或float64。
    - **pool_size** (int|list|tuple) - 池化核的大小。如果它是一个元组或列表，那么它包含三个整数值，(pool_size_Depth, pool_size_Height, pool_size_Width)。若为一个整数，则它的立方值将作为池化核大小，比如若pool_size=2, 则池化核大小为2x2x2。
    - **pool_type** (str) - 池化类型，可以为"max"或"avg"，"max" 对应max-pooling, "avg" 对应average-pooling。
    - **pool_stride** (int|list|tuple) - 池化层的步长。如果它是一个元组或列表，那么它包含三个整数值，(pool_stride_Depth, pool_stride_Height, pool_stride_Width)。若为一个整数，则表示D, H和W维度上stride均为该值。
    - **pool_padding** (int|list|tuple|str) - 池化填充。如果它是一个字符串，可以是"VALID"或者"SAME"，表示填充算法。如果它是一个元组或列表，它可以有3种格式：(1)包含3个整数值：[pad_depth, pad_height, pad_width]；(2)包含6个整数值：[pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]；(3)包含5个二元组：当 ``data_format`` 为"NCDHW"时为[[0,0], [0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]，当 ``data_format`` 为"NDHWC"时为[[0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]。若为一个整数，则表示D、H和W维度上padding均为该值。默认值：0。
    - **global_pooling** （bool）- 是否用全局池化。如果global_pooling = True， ``pool_size`` 和 ``pool_padding`` 将被忽略。默认值：False。
    - **use_cudnn** （bool）- 只在cudnn核中用，需要下载cudnn。默认值：True。
    - **ceil_mode** （bool）- 是否用ceil函数计算输出高度和宽度。如果设为False，则使用floor函数。默认值：False。
    - **name** (str，可选) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值：None。
    - **exclusive** (bool) - 是否在平均池化模式忽略填充值。默认值：True。
    - **data_format** (str) - 输入和输出的数据格式，可以是"NCHW"和"NHWC"。N为批尺寸，C是通道数，H是特征高度，W是特征宽度。默认值："NCHW"。

返回： 池化结果张量，数据类型与 ``input`` 一致。

返回类型：Variable。

抛出异常：
    - ``ValueError`` - 如果 ``pool_type`` 既不是"max"也不是"avg"。
    - ``ValueError`` - 如果 ``global_pooling`` 为False并且 ``pool_size`` 为-1。
    - ``ValueError`` - 如果 ``use_cudnn`` 不是bool值。
    - ``ValueError`` - 如果 ``data_format`` 既不是"NCHW"也不是"NHWC"。
    - ``ValueError`` - 如果 ``pool_padding`` 是字符串，既不是"SAME"也不是"VALID"。
    - ``ValueError`` - 如果 ``pool_padding`` 含有5个二元组，与批尺寸对应的值不为0或者与通道对应的值不为0。


**代码示例**

.. code-block:: python

    # max pool3d
    import paddle.fluid as fluid
    data = fluid.layers.data(
        name='data', shape=[3, 32, 32, 32], dtype='float32')
    pool3d = fluid.layers.pool3d(
                      input=data,
                      pool_size=2,
                      pool_type='max',
                      pool_stride=1,
                      global_pooling=False)

    # average pool3d
    import paddle.fluid as fluid
    data = fluid.layers.data(
        name='data', shape=[3, 32, 32, 32], dtype='float32')
    pool3d = fluid.layers.pool3d(
                      input=data,
                      pool_size=2,
                      pool_type='avg',
                      pool_stride=1,
                      global_pooling=False)

    # global average pool3d
    import paddle.fluid as fluid
    data = fluid.layers.data(
        name='data', shape=[3, 32, 32, 32], dtype='float32')
    pool3d = fluid.layers.pool3d(
                      input=data,
                      pool_size=2,
                      pool_type='avg',
                      pool_stride=1,
                      global_pooling=True)










