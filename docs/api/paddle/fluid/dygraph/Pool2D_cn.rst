.. _cn_api_fluid_dygraph_Pool2D:

Pool2D
-------------------------------

.. py:class:: paddle.fluid.dygraph.Pool2D(pool_size=-1, pool_type='max', pool_stride=1, pool_padding=0, global_pooling=False, use_cudnn=True, ceil_mode=False, exclusive=True, data_format="NCHW")




该接口用于构建 ``Pool2D`` 类的一个可调用对象，具体用法参照 ``代码示例``。其将在神经网络中构建一个二维池化层，并使用上述输入参数的池化配置，为二维空间池化操作，根据 ``input``，池化类型 ``pool_type``，池化核大小 ``pool_size``，步长 ``pool_stride``，填充 ``pool_padding`` 这些参数得到输出。

输入X和输出Out默认是NCHW格式，N为批大小，C是通道数，H是特征高度，W是特征宽度。参数（ ``ksize``, ``strides``, ``paddings`` ）含有两个整型元素。分别表示高度和宽度上的参数。输入X的大小和输出Out的大小可能不一致。

例如：

输入：
    X shape：:math:`\left ( N,C,H_{in},W_{in} \right )`

输出：
    Out shape：:math:`\left ( N,C,H_{out},W_{out} \right )`

如果 ``ceil_mode`` = false：

.. math::
    H_{out} = \frac{(H_{in} - ksize[0] + 2 * paddings[0])}{strides[0]} + 1

.. math::
    W_{out} = \frac{(W_{in} - ksize[1] + 2 * paddings[1])}{strides[1]} + 1

如果 ``ceil_mode`` = true：

.. math::
    H_{out} = \frac{(H_{in} - ksize[0] + 2 * paddings[0] + strides[0] - 1)}{strides[0]} + 1

.. math::
    W_{out} = \frac{(W_{in} - ksize[1] + 2 * paddings[1] + strides[1] - 1)}{strides[1]} + 1

如果 ``exclusive`` = false:

.. math::
    hstart &= i * strides[0] - paddings[0] \\
    hend   &= hstart + ksize[0] \\
    wstart &= j * strides[1] - paddings[1] \\
    wend   &= wstart + ksize[1] \\
    Output(i ,j) &= \frac{sum(Input[hstart:hend, wstart:wend])}{ksize[0] * ksize[1]}

如果 ``exclusive`` = true:

.. math::
    hstart &= max(0, i * strides[0] - paddings[0])\\
    hend &= min(H, hstart + ksize[0]) \\
    wstart &= max(0, j * strides[1] - paddings[1]) \\
    wend & = min(W, wstart + ksize[1]) \\
    Output(i ,j) & = \frac{sum(Input[hstart:hend, wstart:wend])}{(hend - hstart) * (wend - wstart)}

参数
::::::::::::

    - **pool_size** (int|list|tuple，可选) - 池化核的大小。如果它是一个元组或列表，它必须包含两个整数值，(pool_size_Height, pool_size_Width)。若为一个整数，则它的平方值将作为池化核大小，比如若pool_size=2，则池化核大小为2x2。默认值：-1。
    - **pool_type** (str，可选) - 池化类型，可以是”max“对应max-pooling，“avg”对应average-pooling。默认为”max“。
    - **pool_stride** (int|list|tuple，可选)  - 池化层的步长。如果它是一个元组或列表，它将包含两个整数，(pool_stride_Height, pool_stride_Width)。若为一个整数，则表示H和W维度上stride均为该值。默认值为1。
    - **pool_padding** (int|list|tuple，可选) - 填充大小。如果它是一个元组或列表，它必须包含两个整数值，(pool_padding_on_Height, pool_padding_on_Width)。若为一个整数，则表示H和W维度上padding均为该值。默认值为1。
    - **global_pooling** （bool，可选）- 是否用全局池化。如果global_pooling = True， ``pool_size`` 和 ``pool_padding`` 将被忽略，默认False。
    - **use_cudnn** （bool，可选）- 是否用cudnn核，只有已安装cudnn库时才有效。默认True。
    - **ceil_mode** （bool，可选）- 是否用ceil函数计算输出高度和宽度。如果设为False，则使用floor函数。默认为False。
    - **exclusive** (bool，可选) - 是否在平均池化模式忽略填充值。默认为True。
    - **data_format** (str，可选) - 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCHW"和"NHWC"。N是批尺寸，C是通道数，H是特征高度，W是特征宽度。默认值："NCHW"。

返回
::::::::::::
无

抛出异常
::::::::::::

    - ``ValueError`` - 如果 ``pool_type`` 既不是“max”也不是“avg”。
    - ``ValueError`` - 如果 ``global_pooling`` 为False并且 ``pool_size`` 为-1。
    - ``ValueError`` - 如果 ``use_cudnn`` 不是bool值。
    - ``ValueError`` - 如果 ``data_format`` 既不是"NCHW"也不是"NHWC"。

代码示例
::::::::::::

.. code-block:: python

    import paddle.fluid as fluid
    from paddle.fluid.dygraph.base import to_variable
    import numpy as np

    with fluid.dygraph.guard():
       data = np.random.random((3, 32, 32, 5)).astype('float32')
       pool2d = fluid.dygraph.Pool2D(pool_size=2,
                      pool_type='max',
                      pool_stride=1,
                      global_pooling=False)
       pool2d_res = pool2d(to_variable(data))


