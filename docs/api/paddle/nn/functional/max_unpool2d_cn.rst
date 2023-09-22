.. _cn_api_paddle_nn_functional_max_unpool2d:


max_unpool2d
-------------------------------

.. py:function:: paddle.nn.functional.max_unpool2d(x, indices, kernel_size, stride=None,padding=0,data_format="NCHW",output_size=None,name=None)

这个 API 实现了 `2D 最大反池化` 操作

.. note::
   更多细节请参考对应的 `Class` 请参考 :ref:`cn_api_paddle_nn_MaxUnPool2D` 。


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
    - **x** (Tensor) - 形状为 `[N,C,H,W]` 或 `[N,H,W,C]` 的 4-D Tensor，N 是批尺寸，C 是通道数，H 是特征高度，W 是特征宽度，数据类型为 float32 或 float64。
    - **indices** (Tensor) - 形状为 `[N,C,H,W]` 的 4-D Tensor，N 是批尺寸，C 是通道数，H 是特征高度，W 是特征宽度，数据类型为 int32。
    - **kernel_size** (int|list|tuple) - 反池化的滑动窗口大小。
    - **stride** (int|list|tuple，可选) - 池化层的步长。如果它是一个元组或列表，它必须是两个相等的整数，(pool_stride_Height, pool_stride_Width)，默认值：None。
    - **padding** (str|int|list|tuple，可选) - 池化填充，默认值：0。
    - **output_size** (list|tuple，可选) - 目标输出尺寸。如果 output_size 没有被设置，则实际输出尺寸会通过(input_shape, kernel_size, padding)自动计算得出，默认值：None。
    - **data_format** (str，可选) - 输入和输出的数据格式，只能是"NCHW"。N 是批尺寸，C 是通道数，H 是特征高度，W 是特征宽度。默认值："NCHW"
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。



返回
:::::::::

4-D Tensor，unpooling 的计算结果。


代码示例
:::::::::

COPY-FROM: paddle.nn.functional.max_unpool2d
