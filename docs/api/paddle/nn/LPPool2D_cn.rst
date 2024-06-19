
.. _cn_api_paddle_nn_LPPool2D:

LPPool2D
-------------------------------

.. py:function:: paddle.nn.LPPool2D(norm_type, kernel_size, stride=None, padding=0, ceil_mode=False, data_format="NCHW", name=None)
构建 `LPPool2D` 类的一个可调用对象，其将构建一个二维幂平均池化层，根据输入参数 `kernel_size`, `stride`,
`padding` 等参数对输入做平均池化操作。

例如：

输入：

    X 形状：:math:`\left ( N,C,H_{in},W_{in} \right )`

属性：

    - kernel_size: :math:`ksize`
    - stride: :math:`stride`
    - norm_type: :math:`norm\_type`

输出：

    Out 形状：:math:`\left ( N,C,H_{out},W_{out} \right )`

.. math::
    out(N_i, C_j, h, w)  = (\sum_{m=0}^{ksize[0]-1} \sum_{n=0}^{ksize[1]-1}
                               input(N_i, C_j, stride[0] \times h + m, stride[1] \times w + n)^{norm\_type})^{1 / norm\_type}


参数
:::::::::
    - **norm_type** (int|float)：幂平均池化的指数，不可以为 0。
    - **kernel_size** (int|list|tuple)：池化核大小。如果它是一个元组或列表，它必须包含两个整数值，(pool_size_Height, pool_size_Width)。若为一个整数，则它的平方值将作为池化核大小，比如若 pool_size=2，则池化核大小为 2x2。
    - **stride** (int|list|tuple，可选)：池化层的步长。如果它是一个元组或列表，它将包含两个整数，(pool_stride_Height, pool_stride_Width)。若为一个整数，则表示 H 和 W 维度上 stride 均为该值。默认值为 None，这时会使用 kernel_size 作为 stride。
    - **padding** (str|int|list|tuple，可选) 池化填充。如果它是一个字符串，可以是"VALID"或者"SAME"，表示填充算法。如果它是一个元组或列表，它可以有 3 种格式：(1)包含 2 个整数值：[pad_height, pad_width]；(2)包含 4 个整数值：[pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]；(3)包含 4 个二元组：当 data_format 为"NCHW"时为 [[0,0], [0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]，当 data_format 为"NHWC"时为[[0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]。若为一个整数，则表示 H 和 W 维度上均为该值。默认值：0。
    - **ceil_mode** (bool，可选)：是否用 ceil 函数计算输出高度和宽度。如果是 True，则使用 `ceil` 计算输出形状的大小。默认为 False。
    - **data_format** (str，可选)：输入和输出的数据格式，可以是"NCHW"和"NHWC"。N 是批尺寸，C 是通道数，H 是特征高度，W 是特征宽度。默认值："NCHW"
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
:::::::::
    - **x** (Tensor)：默认形状为（批大小，通道数，高度，宽度），即 NCHW 格式的 4-D Tensor。其数据类型为 float16, float32, float64, int32 或 int64。
    - **output** (Tensor)：默认形状为（批大小，通道数，输出特征高度，输出特征宽度），即 NCHW 格式的 4-D Tensor。其数据类型与输入相同。



返回
:::::::::
计算 LPPool2D 的可调用对象



代码示例
:::::::::

COPY-FROM: paddle.nn.LPPool2D
