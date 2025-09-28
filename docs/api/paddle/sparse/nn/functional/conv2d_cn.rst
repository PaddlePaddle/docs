.. _cn_api_paddle_sparse_nn_functional_conv2d:

conv2d
-------------------------------

.. py:function:: paddle.sparse.nn.functional.conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, data_format='NHWC', name=None)

稀疏二维卷积层（sparse convolution2d），根据输入、卷积核、步长（stride）、填充（padding）、空洞大小（dilations）、组参数（groups）计算得到输出特征。输入（Input）和输出（Output）是形状为[N,H,W,C]的多维稀疏坐标格式张量（SparseCooTensors）。其中 N 是批尺寸，H 是特征的高度，W 是特征层宽度，C 是通道数。如果 bias_attr 不为 False，卷积计算会添加偏置项。

对于每个输入 X，计算公式为：

.. math::

    Out = \sigma (W \ast X + b)

其中：

    - :math:`X` ：输入值，NHWC 格式的 Tensor
    - :math:`W` ：卷积核值，HWCM 格式的 Tensor
    - :math:`*` ：卷积操作
    - :math:`b` ：偏置值，1-D Tensor，形为 ``[M]``
    - :math:`Out` ：输出值， ``Out`` 和 ``X`` 的形状可能不同。

参数
::::::::::::
    - **x** (Tensor) - 输入是形状为 [N, H, W, C] 的 4-D SparseCooTensor，输入的数据类型是 float16 或 float32 或 float64。
    - **weight** (Tensor) - 卷积核，形状为 [kH, kW, C/g, M] 的张量，其中 M 是滤波器数（输出通道数），g 是分组数，kD、kH、kW 分别是滤波器的高度和宽度。
    - **bias** (Tensor，可选) - 偏置，形状为 [M] 的张量。
    - **stride** (int|list|tuple，可选) - 步长大小。指的是卷积中的步长。如果步长是列表/元组，则必须包含两个整数（stride_height, stride_width）。否则，stride_height = stride_width = stride。默认：stride = 1。
    - **padding** (string|int|list|tuple，可选) - 填充大小。指的是每个维度两边的零填充数量。如果 `padding` 是字符串，可以是 'VALID' 或 'SAME'，这是填充算法。如果填充大小是元组或列表，可以是以下三种形式：`[pad_height, pad_width]` 或 `[pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`。当 `data_format` 为 `"NHWC"` 时，`padding` 可以是 `[[0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]` 的形式。默认：padding = 0。
    - **dilation** (int|list|tuple，可选) -  空洞大小。空洞卷积时会使用该参数，卷积核对输入进行卷积时，感受野里每相邻两个特征点之间的空洞信息。如果空洞大小是列表或元组，则必须包含两个整数（dilation_height, dilation_width）。否则，dilation_height = dilation_width = dilation。默认：dilation = 1。
    - **groups** (int，可选) - 二维卷积层的组数。根据 Alex Krizhevsky 的深度卷积神经网络（CNN）论文中的成组卷积：当 group=2 时，滤波器的前半部分只与输入通道的前半部分相连，而滤波器的后半部分只与输入通道的后半部分相连。默认：groups=1。目前，仅支持 groups=1。
    - **data_format** (str，可选) - 指定输入的数据格式，输出的数据格式将与输入的一致。可选字符串："NHWC"。默认为 "NHWC"。当为 "NHWC" 时，数据按以下顺序存储：`[batch_size, input_height, input_width, input_channels]`。
    - **name** (str，可选) - 具体用法请参阅 :ref:`api_guide_Name`。通常无需设置名称，默认为 None。

返回
::::::::::::
    - 表示 conv2d 的 SparseCooTensor，其数据类型与输入相同。

返回类型
::::::::::::
Tensor。

代码示例
::::::::::::

COPY-FROM: paddle.sparse.nn.functional.conv2d
