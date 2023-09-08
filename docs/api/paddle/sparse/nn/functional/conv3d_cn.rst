.. _cn_api_paddle_sparse_nn_functional_conv3d:

conv3d
-------------------------------

.. py:function:: paddle.sparse.nn.functional.conv3d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, data_format="NDHWC", name=None)

稀疏三维卷积层（convolution3D layer），根据输入、卷积核、步长（stride）、填充（padding）、空洞大小（dilations）一组参数计算得到输出特征层大小。输入和输出是 NCDHW 或 NDHWC 格式，其中 N 是批尺寸，C 是通道数，D 是特征层深度，H 是特征层高度，W 是特征层宽度。如果 bias_attr 不为 False，卷积计算会添加偏置项。

对每个输入 X ，有等式：

.. math::

    Out = W * X + b

其中：

    - :math:`X` ：输入值，NCDHW 或 NDHWC 格式的 5-D Tensor
    - :math:`W` ：卷积核值，MCDHW 格式的 5-D Tensor
    - :math:`*` ：卷积操作
    - :math:`b` ：偏置值，1-D Tensor，形为 ``[M]``
    - :math:`Out` ：输出值，NCDHW 或 NDHWC 格式的 5-D Tensor，和 ``X`` 的形状可能不同

**示例**

- 输入：

  输入形状：:math:`(N, D_{in}, H_{in}, W_{in}, C_{in})`

  卷积核形状：:math:`(D_f, H_f, W_f, C_{in}, C_{out})`

- 输出：

  输出形状：:math:`(N, D_{out}, H_{out}, W_{out}, C_{out})`

参数
::::::::::::

    - **x** (Tensor) - 输入是形状为 :math:`[N, D, H, W, C]` 的 5-D SparseCooTensor，N 是批尺寸，C 是通道数，D 是特征层深度，H 是特征高度，W 是特征宽度，数据类型为 float16, float32 或 float64 。
    - **weight** (Tensor) - 形状为 :math:`[kD, kH, kW, C/g, M]` 的卷积核（卷积核）。 M 是输出通道数，g 是分组的个数，kH 是卷积核的高度，kW 是卷积核的宽度。
    - **bias** (Tensor，可选) - 偏置项，形状为：:math:`[M]` 。
    - **stride** (int|list|tuple，可选) - 步长大小。卷积核和输入进行卷积计算时滑动的步长。

        - 如果它是一个列表或元组，则必须包含三个整型数：（stride_depth, stride_height,stride_width）。
        - 若为一个整数，stride_depth = stride_height = stride_width = stride。默认值：1。
    - **padding** (int|list|tuple|str，可选) - 填充大小。如果它是一个字符串，可以是"VALID"或者"SAME"，表示填充算法，计算细节可参考上述 ``padding`` = "SAME"或  ``padding`` = "VALID" 时的计算公式。如果它是一个元组或列表，它可以有 3 种格式：

        - (1)包含 5 个二元组：当 ``data_format`` 为"NCDHW"时为 [[0,0], [0,0], [padding_depth_front, padding_depth_back], [padding_height_top, padding_height_bottom], [padding_width_left, padding_width_right]]，当 ``data_format`` 为"NDHWC"时为[[0,0], [padding_depth_front, padding_depth_back], [padding_height_top, padding_height_bottom], [padding_width_left, padding_width_right], [0,0]]；
        - (2)包含 6 个整数值：[padding_depth_front, padding_depth_back, padding_height_top, padding_height_bottom, padding_width_left, padding_width_right]；
        - (3)包含 3 个整数值：[padding_depth, padding_height, padding_width]，此时 padding_depth_front = padding_depth_back = padding_depth, padding_height_top = padding_height_bottom = padding_height, padding_width_left = padding_width_right = padding_width。若为一个整数，padding_depth = padding_height = padding_width = padding。默认值：0。
    - **dilation** (int|list|tuple，可选) - 空洞大小。空洞卷积时会使用该参数，卷积核对输入进行卷积时，感受野里每相邻两个特征点之间的空洞信息。如果空洞大小为列表或元组，则必须包含两个整型数：（dilation_height,dilation_width）。若为一个整数，dilation_height = dilation_width = dilation。默认值：1。
    - **groups** (int，可选) - 二维卷积层的组数。根据 Alex Krizhevsky 的深度卷积神经网络（CNN）论文中的成组卷积：当 group=n，输入和卷积核分别根据通道数量平均分为 n 组，第一组卷积核和第一组输入进行卷积计算，第二组卷积核和第二组输入进行卷积计算，……，第 n 组卷积核和第 n 组输入进行卷积计算。默认值：1。
    - **weight_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值为 None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
    - **bias_attr** （ParamAttr|bool，可选）- 指定偏置参数属性的对象。若 ``bias_attr`` 为 bool 类型，只支持为 False，表示没有偏置参数。默认值为 None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
    - **data_format** (str，可选) - 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCDHW"和"NDHWC"。N 是批尺寸，C 是通道数，D 是特征层深度，H 是特征高度，W 是特征宽度。默认值："NCDHW"。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为 None。

返回
::::::::::::
5-D Tensor ，数据类型与 ``input`` 一致。返回卷积计算的结果。

返回类型
::::::::::::
Tensor。

代码示例
::::::::::::

COPY-FROM: paddle.sparse.nn.functional.conv3d
