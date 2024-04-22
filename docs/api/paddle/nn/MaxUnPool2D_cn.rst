.. _cn_api_paddle_nn_MaxUnPool2D:

MaxUnPool2D
-------------------------------

.. py:class:: paddle.nn.MaxUnPool2D(kernel_size, stride=None,padding=0,data_format="NCHW",output_size=None,name=None)

构建 `MaxUnPool2D` 类的一个可调用对象，根据输入的 input 和最大值位置计算出池化的逆结果。所有非最大值设置为零。

输入：
    X 形状：:math:`(N, C, H_{in}, W_{in})`
输出：
    Output 形状：:math:`(N, C, H_{out}, W_{out})` 具体计算公式为

.. math::
  H_{out} = (H_{in} - 1) \times \text{stride[0]} - 2 \times \text{padding[0]} + \text{kernel_size[0]}

.. math::
  W_{out} = (W_{in} - 1) \times \text{stride[1]} - 2 \times \text{padding[1]} + \text{kernel_size[1]}

或由参数 `output_size` 直接指定



参数
:::::::::
    - **kernel_size** (int|list|tuple) - 反池化的滑动窗口大小。
    - **stride** (int|list|tuple，可选) - 池化层的步长。如果它是一个元组或列表，它必须是两个相等的整数，(pool_stride_Height, pool_stride_Width)，默认值：None。
    - **padding** (str|int|list|tuple，可选) - 池化填充，默认值：0。
    - **output_size** (list|tuple，可选) - 目标输出尺寸。如果 output_size 没有被设置，则实际输出尺寸会通过(input_shape, kernel_size, padding)自动计算得出，默认值：None。
    - **data_format** (str，可选) - 输入和输出的数据格式，只能是"NCHW"。N 是批尺寸，C 是通道数，H 是特征高度，W 是特征宽度。默认值："NCHW"
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。



形状
:::::::::
    - **x** (Tensor)：默认形状为（批大小，通道数，高度，宽度），即 NCHW 格式的 4-D Tensor。其数据类型为 float32， float64 或 int64。
    - **indices** (Tensor)：默认形状为（批大小，通道数，输出特征高度，输出特征宽度），即 NCHW 格式的 4-D Tensor。其数据类型为 int32。
    - **output** (Tensor)：默认形状为（批大小，通道数，输出特征高度，输出特征宽度），即 NCHW 格式的 4-D Tensor。其数据类型与输入一致。


返回
:::::::::
计算 MaxUnPool2D 的可调用对象


代码示例
:::::::::

COPY-FROM: paddle.nn.MaxUnPool2D
