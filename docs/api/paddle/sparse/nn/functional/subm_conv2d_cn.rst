.. _cn_api_paddle_sparse_nn_functional_subm_conv2d_cn:

subm_conv2d_cn
-------------------------------

.. py:function:: paddle.sparse.nn.functional.subm_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, data_format='NHWC', key=None, name=None)

稀疏子流形二维卷积函数根据输入滤波器计算输出以及步幅、填充、扩张、组参数。
输入(Input)和输出(Output)是多维稀疏张量(SparseCooTensors), 其形状为 :math: `[N, H, W, C]` 。
其中 N 是批次大小, C 是通道数, H 是特征的高度, W 是特征的宽度。
如果提供了偏差归因，则将偏差添加到卷积的输出中。

对于每一个输入 :math: `X`, 其计算公式为:

..  math::
    Out = \sigma (W \ast X + b)

在上面的等式中:

    * :math:`X`: 输入值, NHWC 格式的张量。
    * :math:`W`: 筛选值, NHWC 格式的张量。
    * :math:`\\ast`:子流形卷积操作，参考论文: https://arxiv.org/abs/1706.01307.
    * :math:`b`: Bias value, 形状为[M]的一维张量.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.


参数
::::::::::

    - x (Tensor): 输入是形状为 [N, H, W, C] 的四维稀疏张量, 输入数据类型为 float16、float32 或 float64。
    - weight (Tensor): 形状为 [kH, kW, C/g, M] 的卷积核,
                       其中 M 是滤波器(输出通道)的数量, g 是组的数量, kD、kH、kW 分别是滤波器的高度和宽度。
    - bias (Tensor, optional): 偏差, 形状为 [M] 的张量。
    - stride (int|list|tuple, optional): 步长大小, 意味着卷积的步长。如果步幅为 list/tuple, 它必须包含两个整数 (stride_height, stride_width)。
                                         否则, stride_height = stride_width = stride。stride 的默认值为 1。
    - padding (string|int|list|tuple, optional): 填充大小。它表示零填充在每个维度的两侧的数量。
                                                 如果 'padding' 是字符串，则 'VALID' 或 'SAME' 是填充算法。
                                                 如果填充大小是元组或列表，它可以有三种形式：'[pad_heigh, pad_width]' 或 '[pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]',
                                                 当 'data_format' 为 'NHWC' 时, 'padding' 可以采用以下形式
                                                 '[[0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]'。
                                                 padding 的默认值为 0。
    - dilation (int|list|tuple, optional): 扩张大小。它表示内核点之间的间距。
                                           如果 dilation 是列表/元组，则它必须包含两个整数 (dilation_height、dilation_width)。否则, dilation_height = dilation_width = dilation。
                                           dilation 的默认值为 1。
    - groups (int, optional): 二维卷积层的组号。根据 Alex Krizhevsky 的 Deep CNN 论文中的卷积分组：
                              当 group=2 时，滤波器的前半部分仅连接到前半部分的输入通道，而滤波器的后半部分仅连接到输入通道的后半部分。
                              groups 的默认值为 1。目前, 只有 support groups=1。
    - data_format (str, optional): 指定输入的数据格式和输出的数据格式将与输入一致。来自 `"NHWC"` 的可选字符串。默认值为 `"NHWC"`。
                                   当它是 `"NHWC"` 时, 数据按以下顺序存储：`[batch_size, input_height, input_width, input_channels]`。
    - key(str, optional):用于保存或使用相同规则手册的密钥，
                         规则手册的定义和作用是指 https://pdfs.semanticscholar.org/5125/a16039cabc6320c908a4764f32596e018ad3.pdf。这
                         默认值为 None。
    - name(str, optional):有关详细信息，请参阅到 :ref:`api_guide_Name`。
                          通常名称是不需要设置的, 并且默认情况下为空。


返回
::::::::::

    - 表示二维卷积的多维稀疏张量(SparseCooTenstor), 其数据类型与输入相同。


代码示例
::::::::::

COPY-FROM: paddle.sparse.nn.functional.subm_conv2d
