.. _cn_api_fluid_layers_pool3d:

pool3d
-------------------------------

.. py:function:: paddle.fluid.layers.pool3d(input, pool_size=-1, pool_type='max', pool_stride=1, pool_padding=0, global_pooling=False, use_cudnn=True, ceil_mode=False, name=None, exclusive=True)

该OP使用上述输入参数的池化配置，为三维空间池化操作，根据 ``input`` ，池化类型 ``pool_type`` ，池化核大小 ``pool_size`` ，步长 ``pool_stride`` 和填充 ``pool_padding`` 参数计算输出。 输入（X）和输出（Out）采用NCDHW格式，其中N是批大小，C是通道数，D，H和W分别是特征的深度，高度和宽度。 参数（ ``ksize`` ，``strides`` ，``paddings`` ）含有三个整型元素。 分别代表深度，高度和宽度上的参数。 输入（X）大小和输出（Out）大小可能不同。


例如，

输入X形为 :math:`(N, C, D_{in}, H_{in}, W_{in})` ，输出形为 :math:`(N, C, D_{out}, H_{out}, W_{out})`

当ceil_mode = false时，

.. math::

    D_{out} &= \frac{(D_{in} - ksize[0] + 2 * paddings[0])}{strides[0]} + 1\\
    H_{out} &= \frac{(H_{in} - ksize[1] + 2 * paddings[1])}{strides[2]} + 1\\
    W_{out} &= \frac{(W_{in} - ksize[2] + 2 * paddings[2])}{strides[2]} + 1

当ceil_mode = true时，

.. math::

    D_{out} &= \frac{(D_{in} - ksize[0] + 2 * paddings[0] + strides[0] -1)}{strides[0]} + 1\\
    H_{out} &= \frac{(H_{in} - ksize[1] + 2 * paddings[1] + strides[1] -1)}{strides[1]} + 1\\
    W_{out} &= \frac{(W_{in} - ksize[2] + 2 * paddings[2] + strides[2] -1)}{strides[2]} + 1

当exclusive = false时，

.. math::

    dstart &= i * strides[0] - paddings[0]\\
    dend &= dstart + ksize[0]\\
    hstart &= j * strides[1] - paddings[1]\\
    hend &= hstart + ksize[1]\\
    wstart &= k * strides[2] - paddings[2]\\
    wend &= wstart + ksize[2]\\
    Output(i ,j, k) &= \frac{sum(Input[dstart:dend, hstart:hend, wstart:wend])}{ksize[0] * ksize[1] * ksize[2]}



当exclusive = true时，

.. math::

    dstart &= max(0, i * strides[0] - paddings[0])\\
    dend &= min(D, dstart + ksize[0])\\
    hstart &= max(0, j * strides[1] - paddings[1])\\
    hend &= min(H, hstart + ksize[1])\\
    wstart &= max(0, k * strides[2] - paddings[2])\\
    wend &= min(W, wstart + ksize[2])\\
    Output(i ,j, k) &= \frac{sum(Input[dstart:dend, hstart:hend, wstart:wend])}{(dend - dstart) * (hend - hstart) * (wend - wstart)}


参数：
    - **input** (Vairable) - 池化运算的输入张量, 维度为 :math:`[N, C, D, H, W]` 的5-D Tensor。输入张量的格式为NCDHW, N是批尺寸，C是通道数，D是特征深度，H是特征高度，W是特征宽度，数据类型为float32或float64。
    - **pool_size** (int|list|tuple) - 池化窗口的大小。如果为元组类型，那么它应该是由三个整数组成：深度，高度，宽度。若为一个整数，则它的立方值将作为池化核大小，比如若pool_size=2, 则池化核大小为2x2x2。
    - **pool_type** (str) - 池化类型， "max" 对应max-pooling, "avg" 对应average-pooling。
    - **pool_stride** (int|list|tuple) - 池化跨越步长。如果为元组类型，那么它应该是由三个整数组成：深度，高度，宽度。若为一个整数，则表示D, H和W维度上stride均为该值。
    - **pool_padding** (int|list|tuple) - 填充大小。如果为元组类型，那么它应该是由三个整数组成：深度，高度，宽度。若为一个整数，则表示D, H和W维度上padding均为该值。
    - **global_pooling** (bool) - 是否使用全局池化。如果global_pooling = true, ``pool_size`` 和 ``pool_padding`` 将被忽略, 默认False。
    - **use_cudnn** (bool) - 是否用cudnn核，只有在cudnn库安装时有效, 默认True。
    - **ceil_mode** (bool) - 是否用ceil函数计算输出高度和宽度。默认False。如果设为False，则使用floor函数。
    - **name** (None|str) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。
    - **exclusive** (bool) - 是否在平均池化模式忽略填充值。默认为True。

返回： Variable(Tensor) 池化结果张量

返回类型：变量(Variable)，数据类型与 ``input`` 一致

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










