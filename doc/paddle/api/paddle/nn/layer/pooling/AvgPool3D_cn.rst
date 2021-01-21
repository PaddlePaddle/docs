.. _cn_api_nn_AvgPool3D:

AvgPool3D
-------------------------------

.. py:function:: paddle.nn.AvgPool3D(kernel_size, stride=None, padding=0, ceil_mode=False, exclusive=True, divisor_override=None, data_format="NCDHW", name=None)
该接口用于构建 `AvgPool3D` 类的一个可调用对象，其将构建一个二维平均池化层，根据输入参数 `kernel_size`, `stride`,
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
    - **kernel_size** (int|list|tuple): 池化核大小。如果它是一个元组或列表，它必须包含三个整数值， (pool_size_Depth, pool_size_Height, pool_size_Width)。若为一个整数，则表示D，H和W维度上均为该值，比如若pool_size=2, 则池化核大小为[2,2,2]。
    - **stride** (int|list|tuple)：池化层的步长。如果它是一个元组或列表，它将包含两个整数，(pool_stride_Height, pool_stride_Width)。若为一个整数，则表示H和W维度上stride均为该值。默认值为kernel_size.
    - **padding** (string|int|list|tuple) 池化填充。如果它是一个元组或列表，它可以有3种格式：(1)包含3个整数值：[pad_depth, pad_height, pad_width]；(2)包含6个整数值：[pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]；(3)包含5个二元组：当 data_format 为"NCDHW"时为[[0,0], [0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]，当 data_format 为"NDHWC"时为[[0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]。若为一个整数，则表示D、H和W维度上均为该值。默认值：0。
    - **ceil_mode** (bool)：是否用ceil函数计算输出高度和宽度。如果是True，则使用 `ceil` 计算输出形状的大小。默认为False
    - **exclusive** (bool)： 是否在平均池化模式忽略填充值，默认是 `True`.
    - **divisor_override** (int|float)：如果指定，它将用作除数，否则根据`kernel_size`计算除数。 默认`None`.
    - **data_format** (str)： 输入和输出的数据格式，可以是"NCDHW"和"NDHWC"。N是批尺寸，C是通道数，D是特征深度，H是特征高度，W是特征宽度。默认值："NDCHW"。
    - **name** (str)：函数的名字，默认为None.


形状
:::::::::
    - **x** (Tensor): 默认形状为（批大小，通道数，长度，高度，宽度），即NCDHW格式的5-D Tensor。 其数据类型为float16, float32, float64, int32或int64.
    - **output** (Tensor): 默认形状为（批大小，通道数，输出特征长度，输出特征高度，输出特征宽度），即NCDHW格式的5-D Tensor。 其数据类型与输入相同。



返回
:::::::::
计算AvgPool3D的可调用对象

代码示例
:::::::::

.. code-block:: python

        import paddle
        import paddle.nn as nn

        # avg pool3d
        input = paddle.uniform(shape=[1, 2, 32, 32, 32], dtype='float32', min=-1, max=1)
        AvgPool3D = nn.AvgPool3D(kernel_size=2,
                                 stride=2, padding=0)
        output = AvgPool3D(input)
        # output.shape [1, 2, 3, 16, 16]