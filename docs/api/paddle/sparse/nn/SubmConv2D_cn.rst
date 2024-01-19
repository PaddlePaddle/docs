.. _cn_api_paddle_sparse_nn_SubmConv2D:

SubmConv2D
-------------------------------

.. py:class:: paddle.sparse.nn.SubmConv2D(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', key=None, weight_attr=None, bias_attr=None, data_format='NHWC')

**子流形稀疏二维卷积层**

子流形稀疏二维卷积层(submanifold sparse convolution2d layer)根据输入计算输出，卷积核和步长、填充、空洞大小(dilations)一组参数。
输入(input)和输出(Output)是多维的稀疏张量(Sparse Coo Tensor)，
形状为 :math:[N,H,W,C] 其中 N 是批尺寸，C 是通道，H 是特征高度，W 是特征宽度。
如果提供了 bias_attr，则添加偏置项到卷积的输出。
对于每一个输入 :math:`X`，方程是：

..  math::
    Out = W \ast X + b

其中：

    - :math:`X`: 输入值, NDHWC 格式的 Tencer。
    - :math:`W`: 卷积核值, NDHWC 格式的 Tencer。
    - :math:`\\ast`: 子流形卷积运算, 参考论文: https://arxiv.org/abs/1706.01307。
    - :math:`b`: 偏置值, 形状为[M]的 1-D Tencer。
    - :math:`Out`: 输出值, :math:`Out` 和 :math:`X` 的形状可能不同。

参数
::::::::::::

    - **in_channels** (int): - 输入图像的通道数。
    - **out_channels** (int): - 卷积操作产生的输出通道数。
    - **kernel_size** (int|list|tuple): - 卷积核的大小。可以为单个整数或包含三个整数的元组或列表，分别表示卷积核的深度，高和宽。如果为单个整数，表示卷积核的深度，高和宽都等于该整数。
    - **stride** (int|list|tuple, 可选): - 步长大小。如果 stride 是一个列表/元组，它必须包含两个整数，(stride_H, stride_W)。否则, stride_H = stride_W = stride。默认值为 1。
    - **padding** (int|str|tuple|list, 可选): - 填充大小。应为以下几种格式之一；

        - (1) 如果它是一个字符串，可以是 "VALID" 或者 "SAME"，计算细节可参考上述 ``padding`` = "SAME" 或  ``padding`` = "VALID" 时的计算公式。
        - (2) 如果它是一个整数, 则代表它每个 Spartial 维度(depth, height, width) 被 `padding` 的大小填充为零。
        - (3) 一个 list[int] 或 tuple[int]，其长度是 Spartial 维度的数目，它包含每个 Spartial 维度每侧的填充量。它的形式为 [pad_d1, pad_d2, ...]。
        - (4) 一个 list[int] 或 tuple[int]，其长度为 2 * 部分维数。对于所有局部维度，它的形式为 [pad_before, pad_after, pad_before, pad_after, ...]。
        - (5) 一个整数对的列表或元组。它的形式为 [[pad_before, pad_after], [pad_before, pad_after], ...]。

        请注意，批维度和通道维度也包括在内。每对整数对应于输入维度的填充量。批维度和通道维度中的填充应为[0, 0]或者是(0, 0)默认值为 0。
    - **dilation** (int|list|tuple, 可选): - 空洞大小。如果 dilation 是列表或元组, 则它必须包含两个整数 (dilation_H, dilation_W)。否则, dilation_H = dilation_W = dilation。默认值为 1。
    - **groups** (int, 可选): - 二维卷积层的组号。根据 Alex Krizhevsky 的 Deep CNN 论文中的分组卷积:当 group = 2 时, 卷积核的前半部分仅连接到输入通道的前半部分, 而卷积核的后半部分仅连接到输入通道的后半部分。默认值为 1。
    - **padding_mode** (str, 可选): - ``'zeros'``, ``'reflect'``, ``'replicate'`` 或 ``'circular'``。 目前仅支持 ``'zeros'``。
    - **key** (str, 可选): - key 用于保存或使用相同的规则手册，规则手册的定义和作用是指 https://pdfs.semanticscholar.org/5125/a16039cabc6320c908a4764f32596e018ad3.pdf。默认值为 None。
    - **weight_attr** (ParamAttr, 可选): - conv2d 的可学习参数/权重的参数属性。如果设置为 None 或 ParamAttr 的一个属性，则 conv2d 将创建 ParamAttr 作为 param_attr。 如果设置为 None, 则参数初始化为:math:`Normal(0.0, std)`, 并且 :math:`std` 是:math:`(\frac{2.0 }{filter\_elem\_num})^{0.5}`,默认值为 None。
    - **bias_attr** (ParamAttr|bool, 可选): - conv2d 偏差的参数属性。如果设置为 False, 则不会向输出单位添加任何偏置。如果设置为 None 或 ParamAttr 的一个属性，则 conv2d 将创建 ParamAttr 作为 bias_attr。如果未设置 bias_attr 的初始值设定项,则偏置初始化为零。默认值为 None。
    - **data_format** (str, 可选): 指定输入布局的数据格式。它可以是 "NCHW" 或 "NHWC"。目前仅支持 "NHWC"。

属性
::::::::::::

weight
'''''''''
该层卷积核的可学习权重，类型为 ``Parameter``。

bias
'''''''''
该层的可学习偏置，类型为 ``Parameter``。

形状
::::::::::::

    - 输入: :math:`(N, H_{in}, W_{in}, C_{in})`
    - 权重: :math:`(K_{h}, K_{w}, C_{in}, C_{out})`
    - 偏置: :math:`(C_{out})`
    - 输出: :math:`(N, H_{out}, W_{out}, C_{out})`

    其中

    ..  math::

        H_{out}&= \frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (kernel\_size[0] - 1) + 1))}{strides[0]} + 1

        W_{out}&= \frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (kernel\_size[1] - 1) + 1))}{strides[1]} + 1

代码示例
::::::::::::

COPY-FROM: paddle.sparse.nn.SubmConv2D
