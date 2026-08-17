.. _cn_api_paddle_sparse_nn_functional_subm_conv3d:

subm_conv3d
-------------------------------

.. py:function:: paddle.sparse.nn.functional.subm_conv3d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, data_format="NDHWC", key=None, name=None)

子流形稀疏三维卷积层（convolution3D layer），根据输入、卷积核、步长（stride）、填充（padding）、空洞大小（dilations）一组参数计算得到输出特征层大小。输入和输出均为 NDHWC 格式，其中 N 是批尺寸，C 是通道数，D 是特征层深度，H 是特征层高度，W 是特征层宽度。若提供 ``bias``，卷积计算会添加偏置项。

对每个输入 X，有等式：

.. math::

    Out = W * X + b

其中：

    - :math:`X` ：输入值，NDHWC 格式的 5-D SparseCooTensor
    - :math:`W` ：卷积核值，DHWCM 格式的 5-D Tensor
    - :math:`*` ：卷积操作
    - :math:`b` ：偏置值，1-D Tensor，形为 ``[M]``
    - :math:`Out` ：输出值，NDHWC 格式的 5-D SparseCooTensor，和 ``X`` 的形状可能不同

**示例**

- 输入：

  输入形状：:math:`(N, D_{in}, H_{in}, W_{in}, C_{in})`

  卷积核形状：:math:`(D_f, H_f, W_f, C_{in}, C_{out})`

- 输出：

  输出形状：:math:`(N, D_{out}, H_{out}, W_{out}, C_{out})`

参数
::::::::::::

    - **x** (Tensor) - 输入是形状为 :math:`[N, D, H, W, C]` 的 5-D SparseCooTensor，N 是批尺寸，C 是通道数，D 是特征层深度，H 是特征高度，W 是特征宽度，数据类型为 float16, float32 或 float64。
    - **weight** (Tensor) - 形状为 :math:`[kD, kH, kW, C/g, M]` 的卷积核（卷积核）。 M 是输出通道数，g 是分组的个数，kD、kH、kW 分别是卷积核的深度、高度和宽度。
    - **bias** (Tensor，可选) - 偏置项，形状为：:math:`[M]` 。
    - **stride** (int|list|tuple，可选) - 步长大小。卷积核和输入进行卷积计算时滑动的步长。如果它是一个列表或元组，则必须包含三个整型数：（stride_depth, stride_height,stride_width）。若为一个整数，stride_depth = stride_height = stride_width = stride。默认值：1。
    - **padding** (int|list|tuple|str，可选) - 填充大小。如果它是一个字符串，可以是"VALID"或者"SAME"，表示填充算法。如果它是一个元组或列表，可以有以下 3 种格式：(1)包含 5 个二元组：``[[0, 0], [padding_depth_front, padding_depth_back], [padding_height_top, padding_height_bottom], [padding_width_left, padding_width_right], [0, 0]]``；(2)包含 6 个整数值：``[padding_depth_front, padding_depth_back, padding_height_top, padding_height_bottom, padding_width_left, padding_width_right]``；(3)包含 3 个整数值：``[padding_depth, padding_height, padding_width]``。若为一个整数，则深度、高度和宽度方向均使用该值。默认值：0。
    - **dilation** (int|list|tuple，可选) - 空洞大小。若为列表或元组，必须包含三个整数 ``(dilation_depth, dilation_height, dilation_width)``；若为一个整数，则深度、高度和宽度方向均使用该值。默认值：1。
    - **groups** (int，可选) - 卷积层的组数。当前仅支持 ``groups=1``。默认值：1。
    - **data_format** (str，可选) - 指定输入和输出的数据格式。当前仅支持 ``"NDHWC"``，即 ``[batch_size, input_depth, input_height, input_width, input_channels]``。默认值：``"NDHWC"``。
    - **key** (str|None，可选) - 用来保存或使用相同的 rulebook。默认值为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为 None 。

返回
::::::::::::
5-D SparseCooTensor，数据类型与 ``x`` 一致。返回卷积计算的结果。

返回类型
::::::::::::
SparseCooTensor。

代码示例
::::::::::::

COPY-FROM: paddle.sparse.nn.functional.subm_conv3d
