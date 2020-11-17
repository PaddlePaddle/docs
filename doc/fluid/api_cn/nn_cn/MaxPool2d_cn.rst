.. _cn_api_nn_MaxPool2d:

MaxPool2D
-------------------------------

.. py:function:: paddle.nn.MaxPool2D(kernel_size, stride=None, padding=0, ceil_mode=False, return_indices=False, data_format="NCHW", name=None)
该接口用于构建 `MaxPool2D` 类的一个可调用对象，其将构建一个二维平均池化层，根据输入参数 `kernel_size`, `stride`,
`padding` 等参数对输入做最大池化操作。

例如：

输入：
    X 形状：:math:`\left ( N,C,H_{in},W_{in} \right )`
属性：
    kernel_size: :math:`ksize`
    stride: :math:`stride`
输出：
    Out 形状：:math:`\left ( N,C,H_{out},W_{out} \right )`
.. math::
    out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, ksize[0] -1} \max_{n=0, \ldots, ksize[1]-1} \\
                                    & \text{input}(N_i, C_j, \text{stride[0]} \times h + m,
                                                   \text{stride[1]} \times w + n)


参数
:::::::::
    - **kernel_size** (int|list|tuple): 池化核大小。如果它是一个元组或列表，它必须包含两个整数值， (pool_size_Height, pool_size_Width)。若为一个整数，则它的平方值将作为池化核大小，比如若pool_size=2, 则池化核大小为2x2。
    - **stride** (int|list|tuple)：池化层的步长。如果它是一个元组或列表，它将包含两个整数，(pool_stride_Height, pool_stride_Width)。若为一个整数，则表示H和W维度上stride均为该值。默认值为kernel_size.
    - **padding** (string|int|list|tuple) 池化填充。如果它是一个字符串，可以是"VALID"或者"SAME"，表示填充算法。如果它是一个元组或列表，它可以有3种格式：(1)包含2个整数值：[pad_height, pad_width]；(2)包含4个整数值：[pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]；(3)包含4个二元组：当 data_format 为"NCHW"时为 [[0,0], [0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]，当 data_format 为"NHWC"时为[[0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]。若为一个整数，则表示H和W维度上均为该值。默认值：0。
    - **ceil_mode** (bool)：是否用ceil函数计算输出高度和宽度。如果是True，则使用 `ceil` 计算输出形状的大小。
    - **return_indices** (bool)：是否返回最大索引和输出。默认为False.
    - **data_format** (str)： 输入和输出的数据格式，可以是"NCHW"和"NHWC"。N是批尺寸，C是通道数，H是特征高度，W是特征宽度。默认值："NCHW"
    - **name** (str)：函数的名字，默认为None.


返回
:::::::::
计算MaxPool2D的可调用对象

抛出异常
:::::::::
    - ``ValueError`` - 如果 ``padding`` 是一个字符串，但不是["SAME", "VALID"]的其中一个。
    - ``ValueError`` - 如果 ``padding`` 设置为"VALID" 但是"ceil_mode"设置为True
    - ``ShapeError`` - 如果池化后输出的形状小于0。

代码示例
:::::::::

.. code-block:: python

        import paddle
        import paddle.nn as nn
        import numpy as np

        # max pool2d
        input = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32, 32]).astype(np.float32))
        MaxPool2D = nn.MaxPool2D(kernel_size=2,
                                 stride=2, padding=0)
        output = MaxPool2D(input)
        # output.shape [1, 3, 16, 16]
        # for return_indices=True
        MaxPool2D = nn.MaxPool2D(kernel_size=2,stride=2, padding=0, return_indices=True)
        output, max_indices = MaxPool2D(input)
        # output.shape [1, 3, 16, 16], max_indices.shape [1, 3, 16, 16],