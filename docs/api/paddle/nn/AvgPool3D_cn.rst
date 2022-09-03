.. _cn_api_nn_AvgPool3D:

AvgPool3D
-------------------------------

.. py:function:: paddle.nn.AvgPool3D(kernel_size, stride=None, padding=0, ceil_mode=False, exclusive=True, divisor_override=None, data_format="NCDHW", name=None)
构建 `AvgPool3D` 类的一个可调用对象，其将构建一个二维平均池化层，根据输入参数 `kernel_size`, `stride`,
`padding` 等参数对输入做平均池化操作。

例如：

输入：

    X 形状：:math:`\left ( N,C, D_{in}, H_{in},W_{in} \right )`

属性：

    kernel_size: :math:`[KD, KH, KW]`
    stride: :math:`stride`

输出：

    Out 形状：:math:`\left ( N,C, D_{in}, H_{out},W_{out} \right )`

.. math::
    \text{out}(N_i, C_j, d, h, w) ={} & \sum_{k=0}^{kD-1} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1} \\
                                              & \frac{\text{input}(N_i, C_j, \text{stride}[0] \times d + k,
                                                      \text{stride}[1] \times h + m, \text{stride}[2] \times w + n)}
                                                     {kD \times kH \times kW}


参数
:::::::::
    - **kernel_size** (int|list|tuple)：池化核大小。如果它是一个元组或列表，它必须包含三个整数值，(pool_size_Depth, pool_size_Height, pool_size_Width)。若为一个整数，则表示 D，H 和 W 维度上均为该值，比如若 pool_size=2，则池化核大小为[2,2,2]。
    - **stride** (int|list|tuple，可选)：池化层的步长。如果它是一个元组或列表，它将包含两个整数，(pool_stride_Height, pool_stride_Width)。若为一个整数，则表示 H 和 W 维度上 stride 均为该值。默认值为 None，这时会使用 kernel_size 作为 stride。
    - **padding** (str|int|list|tuple，可选) 池化填充。如果它是一个元组或列表，它可以有 3 种格式：(1)包含 3 个整数值：[pad_depth, pad_height, pad_width]；(2)包含 6 个整数值：[pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]；(3)包含 5 个二元组：当 data_format 为"NCDHW"时为[[0,0], [0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]，当 data_format 为"NDHWC"时为[[0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]。若为一个整数，则表示 D、H 和 W 维度上均为该值。默认值：0。
    - **ceil_mode** (bool，可选)：是否用 ceil 函数计算输出高度和宽度。如果是 True，则使用 `ceil` 计算输出形状的大小。默认为 False
    - **exclusive** (bool，可选)：是否在平均池化模式忽略填充值，默认是 `True`。
    - **divisor_override** (int|float，可选)：如果指定，它将用作除数，否则根据`kernel_size`计算除数。默认`None`。
    - **data_format** (str，可选)：输入和输出的数据格式，可以是"NCDHW"和"NDHWC"。N 是批尺寸，C 是通道数，D 是特征深度，H 是特征高度，W 是特征宽度。默认值："NDCHW"。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


形状
:::::::::
    - **x** (Tensor)：默认形状为（批大小，通道数，长度，高度，宽度），即 NCDHW 格式的 5-D Tensor。其数据类型为 float16, float32, float64, int32 或 int64。
    - **output** (Tensor)：默认形状为（批大小，通道数，输出特征长度，输出特征高度，输出特征宽度），即 NCDHW 格式的 5-D Tensor。其数据类型与输入相同。



返回
:::::::::
计算 AvgPool3D 的可调用对象

代码示例
:::::::::

COPY-FROM: paddle.nn.AvgPool3D
