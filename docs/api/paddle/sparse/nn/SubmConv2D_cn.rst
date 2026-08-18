.. _cn_api_paddle_sparse_nn_SubmConv2D:

SubmConv2D
-------------------------------

.. py:class:: paddle.sparse.nn.SubmConv2D(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', key=None, weight_attr=None, bias_attr=None, data_format='NHWC', backend=None)

**子流形稀疏二维卷积层**

子流形稀疏二维卷积层(submanifold sparse convolution2d layer)根据输入计算输出，卷积核和步长、填充、空洞大小(dilations)一组参数。
输入(input)和输出(Output)是多维的稀疏张量(Sparse Coo Tensor)，
形状为 :math:`[N,H,W,C]` 其中 N 是批尺寸，C 是通道，H 是特征高度，W 是特征宽度。
如果提供了 ``bias_attr``，则添加偏置项到卷积的输出。
对于每一个输入 :math:`X`，方程是：

..  math::
    Out = W \ast X + b

其中：

    - :math:`X` : 输入值, NHWC 格式的 SparseCooTensor。
    - :math:`W` : 卷积核值, HWCM 格式的 Tensor。
    - :math:`\\ast` : 子流形卷积运算, 参考论文: `Submanifold Sparse Convolutional Networks <https://arxiv.org/abs/1706.01307>`_ 。
    - :math:`b` : 偏置值, 形状为[M]的 1-D Tensor。
    - :math:`Out` : 输出值, :math:`Out` 和 :math:`X` 的形状可能不同。

参数
::::::::::::

    - **in_channels** (int) - 输入图像的通道数。
    - **out_channels** (int) - 卷积操作产生的输出通道数。
    - **kernel_size** (int|list|tuple) - 卷积核的大小。可以为单个整数或包含两个整数的元组或列表，分别表示卷积核的高和宽。如果为单个整数，表示卷积核的高和宽都等于该整数。
    - **stride** (int|list|tuple，可选) - 步长大小。如果 stride 是一个列表/元组，它必须包含两个整数，(stride_H, stride_W)。否则, stride_H = stride_W = stride。默认值为 1。
    - **padding** (int|str|tuple|list，可选) - 填充大小。可以是 ``"VALID"`` 或 ``"SAME"``；也可以是一个整数、包含两个整数的 ``[pad_height, pad_width]``、包含四个整数的 ``[pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]``，或四个二元组 ``[[0, 0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0, 0]]``。若为一个整数，则高和宽方向均使用该值。默认值为 0。
    - **dilation** (int|list|tuple，可选) - 空洞大小。如果 dilation 是列表或元组, 则它必须包含两个整数 (dilation_H, dilation_W)。否则, dilation_H = dilation_W = dilation。默认值为 1。
    - **groups** (int，可选) - 卷积层的组数。当前仅支持 ``groups=1``。默认值为 1。
    - **padding_mode** (str，可选) - 当前仅支持 ``'zeros'``。默认值为 ``'zeros'``。
    - **key** (str|None，可选) - key 用于保存或使用相同的规则手册，规则手册的定义和作用是指 https://pdfs.semanticscholar.org/5125/a16039cabc6320c908a4764f32596e018ad3.pdf。默认值为 None。
    - **weight_attr** (ParamAttr，可选) - conv2d 的可学习参数/权重的参数属性。如果设置为 None 或 ParamAttr 的一个属性，则 conv2d 将创建 ParamAttr 作为 param_attr。 如果设置为 None, 则参数初始化为 :math:`Normal(0.0, std)` , 并且 :math:`std` 是 :math:`(\frac{2.0 }{filter\_elem\_num})^{0.5}` ,默认值为 None。
    - **bias_attr** (ParamAttr|bool，可选) - conv2d 偏差的参数属性。如果设置为 False, 则不会向输出单位添加任何偏置。如果设置为 None 或 ParamAttr 的一个属性，则 conv2d 将创建 ParamAttr 作为 bias_attr。如果未设置 bias_attr 的初始值设定项,则偏置初始化为零。默认值为 None。
    - **data_format** (str，可选) - 指定输入布局的数据格式。当前仅支持 "NHWC"。
    - **backend** (str，可选) - 指定稀疏卷积实现后端。可选值为 ``"igemm"`` 或 ``None``，默认值为 None。

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
