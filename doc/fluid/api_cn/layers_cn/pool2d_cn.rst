.. _cn_api_fluid_layers_pool2d:

pool2d
-------------------------------

.. py:function:: paddle.fluid.layers.pool2d(input, pool_size=-1, pool_type='max', pool_stride=1, pool_padding=0, global_pooling=False, use_cudnn=True, ceil_mode=False, name=None, exclusive=True)

pooling2d操作符根据 ``input`` ， 池化类型 ``pool_type`` ， 池化核大小 ``pool_size`` , 步长 ``pool_stride`` ，填充 ``pool_padding`` 这些参数得到输出。

输入X和输出Out是NCHW格式，N为batch尺寸，C是通道数，H是特征高度，W是特征宽度。

参数（ksize,strides,paddings）含有两个元素。这两个元素分别代表高度和宽度。输入X的大小和输出Out的大小可能不一致。

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



参数：
    - **input** (Variable) - 池化操作的输入张量。输入张量格式为NCHW，N为批尺寸，C是通道数，H是特征高度，W是特征宽度
    - **pool_size** (int|list|tuple)  - 池化核的大小。如果它是一个元组或列表，它必须包含两个整数值， (pool_size_Height, pool_size_Width)。若为一个整数，则它的平方值将作为池化核大小，比如若pool_size=2, 则池化核大小为2x2。
    - **pool_type** (string) - 池化类型，可以是“max”对应max-pooling，“avg”对应average-pooling
    - **pool_stride** (int|list|tuple)  - 池化层的步长。如果它是一个元组或列表，它将包含两个整数，(pool_stride_Height, pool_stride_Width)。否则它是一个整数的平方值。
    - **pool_padding** (int|list|tuple) - 填充大小。如果它是一个元组或列表，它必须包含两个整数值，(pool_padding_on_Height, pool_padding_on_Width)。否则它是一个整数的平方值。
    - **global_pooling** （bool，默认false）- 是否用全局池化。如果global_pooling = true， ``pool_size`` 和 ``pool_padding`` 将被忽略。
    - **use_cudnn** （bool，默认false）- 只在cudnn核中用，需要下载cudnn
    - **ceil_mode** （bool，默认false）- 是否用ceil函数计算输出高度和宽度。默认False。如果设为False，则使用floor函数
    - **name** （str|None） - 该层名称（可选）。若设为None，则自动为该层命名。
    - **exclusive** (bool) - 是否在平均池化模式忽略填充值。默认为True。

返回：池化结果

返回类型：变量（Variable）

抛出异常：
    - ``ValueError`` - 如果 ``pool_type`` 既不是“max”也不是“avg”
    - ``ValueError`` - 如果 ``global_pooling`` 为False并且‘pool_size’为-1
    - ``ValueError`` - 如果 ``use_cudnn`` 不是bool值

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(
        name='data', shape=[3, 32, 32], dtype='float32')
    pool2d = fluid.layers.pool2d(
                  input=data,
                  pool_size=2,
                  pool_type='max',
                  pool_stride=1,
                  global_pooling=False)









