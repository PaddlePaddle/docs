.. _cn_api_paddle_nn_functional_avg_pool3d:

avg_pool3d
-------------------------------

.. py:function:: paddle.nn.functional.avg_pool3d(x, kernel_size, stride=None, padding=0, ceil_mode=False, exclusive=True, divisor_override=None, data_format="NCDHW", name=None)
该函数是一个三维平均池化函数，根据输入参数 `kernel_size`, `stride`,
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
    - **x** (Tensor)：形状为 [N,C,D,H,W] 或 [N,D,H,W,C] 的 5-D Tensor，N 是批尺寸，C 是通道数，D 是特征深度，H 是特征高度，W 是特征宽度，数据类型为 float16、float32 或 float64。
    - **kernel_size** (int|list|tuple)：池化核大小。如果它是一个元组或列表，它必须包含三个整数值，(kernel_size_Depth, kernel_size_Height, kernel_size_Width)。若为一个整数，则表示 D，H 和 W 维度上均为该值，比如若 kernel_size=2，则池化核大小为[2,2,2]。
    - **stride** (int|list|tuple)：池化层的步长。如果它是一个元组或列表，它将包含两个整数，(pool_stride_Height, pool_stride_Width)。若为一个整数，则表示 H 和 W 维度上 stride 均为该值。默认值为 kernel_size。
    - **padding** (string|int|list|tuple) 池化填充。如果它是一个元组或列表，它可以有 3 种格式：(1)包含 3 个整数值：[pad_depth, pad_height, pad_width]；(2)包含 6 个整数值：[pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]；(3)包含 5 个二元组：当 data_format 为"NCDHW"时为[[0,0], [0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]，当 data_format 为"NDHWC"时为[[0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]。若为一个整数，则表示 D、H 和 W 维度上均为该值。默认值：0。
    - **ceil_mode** (bool)：是否用 ceil 函数计算输出高度和宽度。如果是 True，则使用 `ceil` 计算输出形状的大小。默认为 False
    - **exclusive** (bool)：是否在平均池化模式忽略填充值，默认是 `True`。
    - **divisor_override** (int|float)：如果指定，它将用作除数，否则根据`kernel_size`计算除数。默认`None`。
    - **data_format** (str)：输入和输出的数据格式，可以是"NCDHW"和"NDHWC"。N 是批尺寸，C 是通道数，D 是特征深度，H 是特征高度，W 是特征宽度。默认值："NDCHW"。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。





返回
:::::::::
5-D Tensor，数据类型与输入 x 一致。


代码示例
:::::::::

COPY-FROM: paddle.nn.functional.avg_pool3d
