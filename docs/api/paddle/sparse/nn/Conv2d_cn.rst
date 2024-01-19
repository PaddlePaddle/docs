.. _cn_api_paddle_sparse_nn_Conv2D:

Conv2D
-------------------------------

.. py:class:: paddle.sparse.nn.Conv2D(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', weight_attr=None, bias_attr=None, data_format="NHWC")

**稀疏二维卷积层**

二维稀疏卷积层（Sparse convolution2d layer）根据输入、卷积核、步长（stride）、填充（padding）、空洞大小（dilations）、组参数（groups）计算得到输出特征。

输入（Input）和输出（Output）是形状为[N,H,W,C]的多维稀疏坐标格式张量（SparseCooTensors）。

其中 N 是批量大小，C 是通道数，H 是特征的高度，W 是特征层宽度。如果 bias_attr 不为 False，卷积计算会添加偏置项。

对于每个输入 X，计算公式为：

.. math::

    Out = W \ast X + b

其中：

    - :math:`X`：输入值，NHWC 格式的 Tensor
    - :math:`W`：卷积核值，HWCM 格式的 Tensor
    - :math:`*`：卷积操作
    - :math:`b`：偏置值，1-D Tensor，形为 ``[M]``
    - :math:`Out`：输出值，NHWC 格式的 Tensor，和 ``X`` 的形状可能不同

参数
::::::::::::

    - **in_channels** (int) - 输入图像的通道数。
    - **out_channels** (int) - 由卷积操作产生的输出的通道数。
    - **kernel_size** (int|list|tuple) - 卷积核大小。
    - **stride** (int|list|tuple，可选) - 步长大小。可以为单个整数或包含三个整数的元组或列表，分别表示卷积沿着深度，高和宽的步长。如果为单个整数，表示沿着高和宽的步长都等于该整数。默认值：1。
    - **padding** (int|str|tuple|list，可选) - 填充大小。填充可以是以下形式之一：
    
        1. 字符串 ['valid', 'same']。
        2. 一个整数，表示每个空间维度（高度、宽度）的零填充大小。
        3. 长度为空间维度数量的列表[int]或元组[int]，包含每个空间维度两边的填充量。形式为 [pad_d1, pad_d2, ...]。
        4. 长度为空间维度数量的两倍的列表[int]或元组[int]。形式为 [pad_before, pad_after, pad_before, pad_after, ...]。
        5. 成对整数的列表或元组。形式为 [[pad_before, pad_after], [pad_before, pad_after], ...]。
        
        注意，批量维度和通道维度也包括在内。每对整数对应输入的一个维度的填充量。批量维度和通道维度的填充应为 [0, 0] 或 (0, 0)。默认值为 0。
    - **dilation** (int|list|tuple，可选) - 空洞大小。可以为单个整数或包含三个整数的元组或列表，分别表示卷积核中的元素沿着深度，高和宽的空洞。如果为单个整数，表示深度，高和宽的空洞都等于该整数。默认值：1。
    - **groups** (int，可选) - 二维卷积层的组数。根据 Alex Krizhevsky 的深度卷积神经网络（CNN）论文中的成组卷积：当 group=n，输入和卷积核分别根据通道数量平均分为 n 组，第一组卷积核和第一组输入进行卷积计算，第二组卷积核和第二组输入进行卷积计算，……，第 n 组卷积核和第 n 组输入进行卷积计算。默认值：1。
    - **padding_mode** (str，可选) - 填充模式。包括 ``'zeros'``, ``'reflect'``, ``'replicate'`` 或者 ``'circular'``。默认值：``'zeros'`` 。
    - **weight_attr** (ParamAttr，可选) - conv2d 的可学习参数/权重的参数属性。如果设置为 None 或 ParamAttr 的一个属性，conv2d 将创建 ParamAttr 作为 param_attr。如果设置为 None，则参数初始化为 :math:`Normal(0.0, std)`，:math:`std` 为 :math:`(\frac{2.0 }{filter\_elem\_num})^{0.5}`。默认值为 None。
    - **bias_attr** (ParamAttr|bool，可选) - conv2d 的偏置参数属性。如果设置为 False，则不会在输出单元中添加偏置。如果设置为 None 或 ParamAttr 的一个属性，conv2d 将创建 ParamAttr 作为 bias_attr。如果 bias_attr 的初始化器未设置，则偏置初始化为零。默认值为 None。
    - **data_format** (str，可选) - 指定输入的数据格式。可以是 "NCHW" 或 "NHWC"。目前仅支持 "NHWC"。N 是批尺寸，C 是通道数，D 是特征深度，H 是特征高度，W 是特征宽度。默认值："NDHWC"。 当前只支持"NDHWC"。


属性
::::::::::::

weight
'''''''''
本层的可学习参数，类型为 ``Parameter``

bias
'''''''''
本层的可学习偏置，类型为 ``Parameter``

形状
::::::::::::

    - 输入：:math:`(N, H_{in}, W_{in}, C_{in})`
    - 卷积核：:math:`(K_{h}, K_{w}, C_{in}, C_{out})`
    - 偏置：:math:`(C_{out})`
    - 输出：:math:`(N, H_{out}, W_{out}, C_{out})`

    其中

   .. math::

    H_{out}&= \frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (kernel\_size[0] - 1) + 1))}{strides[0]} + 1

    W_{out}&= \frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (kernel\_size[1] - 1) + 1))}{strides[1]} + 1


代码示例
::::::::::::

COPY-FROM: paddle.sparse.nn.Conv2D
