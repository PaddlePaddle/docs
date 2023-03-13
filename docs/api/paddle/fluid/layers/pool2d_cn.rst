.. _cn_api_fluid_layers_pool2d:

pool2d
-------------------------------

.. py:function:: paddle.fluid.layers.pool2d(input, pool_size=-1, pool_type='max', pool_stride=1, pool_padding=0, global_pooling=False, use_cudnn=True, ceil_mode=False, name=None, exclusive=True, data_format="NCHW")




该 OP 使用上述输入参数的池化配置，为二维空间池化操作，根据 ``input``，池化核大小 ``pool_size``，池化类型 ``pool_type``，步长 ``pool_stride``，填充 ``pool_padding`` 等参数得到输出。

输入 ``input`` 和输出（out）采用 NCHW 或 NHWC 格式，N 为批大小，C 是通道数，H 是特征高度，W 是特征宽度。

参数
::::::::::::
``pool_size`` 和 ``pool_stride`` 含有两个整型元素，分别表示高度和宽度维度上的参数。

输入 ``input`` 和输出（out）的形状可能不同。


例如：

输入：
    ``input`` 的形状：:math:`\left ( N,C,H_{in},W_{in} \right )`

输出：
    ``out`` 的形状：:math:`\left ( N,C,H_{out},W_{out} \right )`

如果 ``ceil_mode`` = false：

.. math::
    H_{out} = \frac{(H_{in} - pool\_size[0] + pad\_height\_top + pad\_height\_bottom)}{pool\_stride[0]} + 1

.. math::
    W_{out} = \frac{(W_{in} - pool\_size[1] + pad\_width\_left + pad\_width\_right)}{pool\_stride[1]} + 1

如果 ``ceil_mode`` = true：

.. math::
    H_{out} = \frac{(H_{in} - pool\_size[0] + pad\_height\_top + pad\_height\_bottom + pool\_stride[0] - 1)}{pool\_stride[0]} + 1

.. math::
    W_{out} = \frac{(W_{in} - pool\_size[1] + pad\_width\_left + pad\_width\_right + pool\_stride[1] - 1)}{pool\_stride[1]} + 1

如果 ``exclusive`` = false:

.. math::
    hstart &= i * pool\_stride[0] - pad\_height\_top \\
    hend   &= hstart + pool\_size[0] \\
    wstart &= j * pool\_stride[1] - pad\_width\_left \\
    wend   &= wstart + pool\_size[1] \\
    Output(i ,j) &= \frac{sum(Input[hstart:hend, wstart:wend])}{pool\_size[0] * pool\_size[1]}

如果 ``exclusive`` = true:

.. math::
    hstart &= max(0, i * pool\_stride[0] - pad\_height\_top) \\
    hend &= min(H, hstart + pool\_size[0]) \\
    wstart &= max(0, j * pool\_stride[1] - pad\_width\_left) \\
    wend & = min(W, wstart + pool\_size[1]) \\
    Output(i ,j) & = \frac{sum(Input[hstart:hend, wstart:wend])}{(hend - hstart) * (wend - wstart)}

如果 ``pool_padding`` = "SAME":

.. math::
    H_{out} = \frac{(H_{in} + pool\_stride[0] - 1)}{pool\_stride[0]}

.. math::
    W_{out} = \frac{(W_{in} + pool\_stride[1] - 1)}{pool\_stride[1]}

如果 ``pool_padding`` = "VALID":

.. math::
    H_{out} = \frac{(H_{in} - pool\_size[0])}{pool\_stride[0]} + 1

.. math::
    W_{out} = \frac{(W_{in} - pool\_size[1])}{pool\_stride[1]} + 1

参数
::::::::::::

    - **input** (Variable) - 形状为 :math:`[N, C, H, W]` 或 :math:`[N, H, W, C]` 的 4-D Tensor，N 是批尺寸，C 是通道数，H 是特征高度，W 是特征宽度，数据类型为 float32 或 float64。
    - **pool_size** (int|list|tuple)  - 池化核的大小。如果它是一个元组或列表，那么它包含两个整数值：(pool_size_Height, pool_size_Width)。若为一个整数，则表示 H 和 W 维度上均为该值，比如若 pool_size=2，则池化核大小为[2,2]。
    - **pool_type** (str) - 池化类型，可以为"max"或"avg"，"max"对应 max-pooling，"avg"对应 average-pooling。默认值："max"。
    - **pool_stride** (int|list|tuple)  - 池化层的步长。如果它是一个元组或列表，它将包含两个整数：(pool_stride_Height, pool_stride_Width)。若为一个整数，则表示 H 和 W 维度上均为该值，比如若 pool_stride=3，则池化层步长为[3,3]。默认值：1。
    - **pool_padding** (int|list|tuple|str) - 池化填充。如果它是一个字符串，可以是"VALID"或者"SAME"，表示填充算法，计算细节可参考上述 ``pool_padding`` = "SAME"或  ``pool_padding`` = "VALID" 时的计算公式。如果它是一个元组或列表，它可以有 3 种格式：(1)包含 2 个整数值：[pad_height, pad_width]；(2)包含 4 个整数值：[pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]；(3)包含 4 个二元组：当 ``data_format`` 为"NCHW"时为 [[0,0], [0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]，当 ``data_format`` 为"NHWC"时为[[0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]。若为一个整数，则表示 H 和 W 维度上均为该值。默认值：0。
    - **global_pooling** （bool）- 是否用全局池化。如果 global_pooling = True，已设置的 ``pool_size`` 和 ``pool_padding`` 会被忽略，``pool_size`` 将被设置为 :math:`[H_{in}, W_{in}]` ， ``pool_padding`` 将被设置为 0。默认值：False。
    - **use_cudnn** （bool）- 是否使用 cudnn 内核。只有已安装 cudnn 库时才有效。默认值：True。
    - **ceil_mode** （bool）- 是否用 ceil 函数计算输出高度和宽度。计算细节可参考上述 ``ceil_mode`` = true 或  ``ceil_mode`` = false 时的计算公式。默认值：False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
    - **exclusive** (bool) - 是否在平均池化模式忽略填充值。计算细节可参考上述 ``exclusive`` = true 或 ``exclusive`` = false 时的计算公式。默认值：True。
    - **data_format** (str) - 输入和输出的数据格式，可以是"NCHW"和"NHWC"。N 是批尺寸，C 是通道数，H 是特征高度，W 是特征宽度。默认值："NCHW"。

返回
::::::::::::
 4-D Tensor，数据类型与 ``input`` 一致。

返回类型
::::::::::::
Variable。

抛出异常
::::::::::::

    - ``ValueError`` - 如果 ``pool_type`` 既不是"max"也不是"avg"。
    - ``ValueError`` - 如果 ``global_pooling`` 为 False 并且 ``pool_size`` 为-1。
    - ``TypeError`` - 如果 ``use_cudnn`` 不是 bool 值。
    - ``ValueError`` - 如果 ``data_format`` 既不是"NCHW"也不是"NHWC"。
    - ``ValueError`` - 如果 ``pool_padding`` 是字符串，既不是"SAME"也不是"VALID"。
    - ``ValueError`` - 如果 ``pool_padding`` 是"VALID"，但是 ``ceil_mode`` 是 True。
    - ``ValueError`` - 如果 ``pool_padding`` 含有 4 个二元组，与批尺寸对应维度的值不为 0 或者与通道对应维度的值不为 0。
    - ``ShapeError`` - 如果 ``input`` 既不是 4-D Tensor 也不是 5-D Tensor。
    - ``ShapeError`` - 如果 ``input`` 的维度减去 ``pool_stride`` 的尺寸大小不是 2。
    - ``ShapeError`` - 如果 ``pool_size`` 和 ``pool_stride`` 的尺寸大小不相等。
    - ``ShapeError`` - 如果计算出的输出形状的元素值不大于 0。


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.pool2d
