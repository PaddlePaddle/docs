.. _cn_api_nn_MaxUnPool1D:

MaxUnPool1D
-------------------------------

.. py:class:: paddle.nn.MaxUnPool1D(kernel_size, stride=None, padding=0, data_format="NCL", output_size=None, name=None)

构建 `MaxUnPool1D` 类的一个可调用对象，根据输入的 input 和最大值位置计算出池化的逆结果。所有非最大值设置为零。

输入：
    X 形状：:math:`(N, C, L_{in})`
输出：
    Output 形状：:math:`(N, C, L_{out})` 具体计算公式为

.. math::
  L_{out} = (L_{in} - 1) \times \text{stride} - 2 \times \text{padding} + \text{kernel_size}

或由参数 `output_size` 直接指定。



参数
:::::::::
    - **kernel_size** (int|list|tuple) - 反池化的滑动窗口大小。
    - **stride** (int|list|tuple，可选) - 池化层的步长。如果它是一个元组或列表，它必须包含一个整数，(pool_stride_Length)，默认值：None。
    - **padding** (str|int|list|tuple，可选) - 池化填充，默认值：0。
    - **output_size** (list|tuple，可选) - 目标输出尺寸。如果 output_size 没有被设置，则实际输出尺寸会通过(input_shape, kernel_size, stride, padding)自动计算得出，默认值：None。
    - **data_format** (str，可选) - 输入和输出的数据格式，只能是"NCL"。N 是批尺寸，C 是通道数，L 是特征长度。默认值："NCL"
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。



形状
:::::::::
    - **x** (Tensor)：默认形状为（批大小，通道数，长度），即 NCL 格式的 3-D Tensor。其数据类型为 float32 或 float64。
    - **indices** (Tensor)：默认形状为（批大小，通道数，输出特征长度），即 NCL 格式的 3-D Tensor。其数据类型为 int32。
    - **output** (Tensor)：默认形状为（批大小，通道数，输出特征长度），即 NCL 格式的 3-D Tensor。其数据类型与输入一致。


返回
:::::::::
计算 MaxUnPool1D 的可调用对象


代码示例
:::::::::
COPY-FROM: paddle.nn.MaxUnPool1D
