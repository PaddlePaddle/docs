.. _cn_api_paddle_vision_ops_DeformConv2D:

DeformConv2D
-------------------------------

.. py:class:: paddle.vision.ops.DeformConv2D(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, deformable_groups=1, groups=1, weight_attr=None, bias_attr=None)


deform_conv2d 对输入 4-D Tensor 计算 2-D 可变形卷积。给定输入 Tensor x，输出 Tensor y，可变形卷积运算如下所示：

可形变卷积 v2:

  :math:`y(p) = \sum_{k=1}^{K}{w_k * x(p + p_k + \Delta p_k) * \Delta m_k}`

可形变卷积 v1:

  :math:`y(p) = \sum_{k=1}^{K}{w_k * x(p + p_k + \Delta p_k)}`

其中 :math:`\Delta p_k` 和 :math:`\Delta m_k` 分别为第 k 个位置的可学习偏移和调制标量。在 deformable conv v1 中 :math:`\Delta m_k` 为 1。

具体细节可以参考论文：`<<Deformable ConvNets v2: More Deformable, Better Results>> <https://arxiv.org/abs/1811.11168v2>`_ 和 `<<Deformable Convolutional Networks>> <https://arxiv.org/abs/1703.06211>`_ 。

**示例**

输入：
    input 形状：:math:`(N, C_{in}, H_{in}, W_{in})`

    卷积核形状：:math:`(C_{out}, C_{in}, H_f, W_f)`

    offset 形状：:math:`(N, 2 * H_f * W_f, H_{out}, W_{out})`

    mask 形状：:math:`(N, H_f * W_f, H_{out}, W_{out})`

输出：
    输出形状：:math:`(N, C_{out}, H_{out}, W_{out})`

其中

.. math::

    H_{out}&= \frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (H_f - 1) + 1))}{strides[0]} + 1

    W_{out}&= \frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (W_f - 1) + 1))}{strides[1]} + 1


参数
::::::::::::

    - **in_channels** (int) - 输入图像的通道数。
    - **out_channels** (int) - 由卷积操作产生的输出的通道数。
    - **kernel_size** (int|list|tuple) - 卷积核大小。可以为单个整数或包含两个整数的元组或列表，分别表示卷积核的高和宽。如果为单个整数，表示卷积核的高和宽都等于该整数。
    - **stride** (int|list|tuple，可选) - 步长大小。可以为单个整数或包含两个整数的元组或列表，分别表示卷积沿着高和宽的步长。如果为单个整数，表示沿着高和宽的步长都等于该整数。默认值：1。
    - **padding** (int|list|tuple，可选) - 填充大小。卷积核操作填充大小。如果它是一个列表或元组，则必须包含两个整型数：（padding_height,padding_width）。若为一个整数，padding_height = padding_width = padding。默认值：0。
    - **dilation** (int|list|tuple，可选) - 空洞大小。可以为单个整数或包含两个整数的元组或列表，分别表示卷积核中的元素沿着高和宽的空洞。如果为单个整数，表示高和宽的空洞都等于该整数。默认值：1。
    - **deformable_groups** (int，可选) - 可变形卷积组数。默认值：1。
    - **groups** (int，可选) - 三维卷积层的组数。根据 Alex Krizhevsky 的深度卷积神经网络（CNN）论文中的分组卷积：当 group=2，前半部分卷积核只和前半部分输入进行卷积计算，后半部分卷积核和后半部分输入进行卷积计算。默认值：1。
    - **weight_attr** (ParamAttr，可选) - 二维卷积层的可学习参数/权重的属性。如果设置为 None 或 ParamAttr，二维卷积层将创建 ParamAttr 作为 param_attr。如果设置为 None，参数将初始化为 :math:`Normal(0.0, std)` ，且 :math:`std` 为
            :math:`(\frac{2.0 }{filter\_elem\_num})^{0.5}` ，默认值为 None 。
    - **bias_attr** (ParamAttr|bool，可选)- 二维卷积层偏置参数属性对象，如果设置为 False，则不会向输出单元添加任何偏差。如果设置为 None 或 ParamAttr，二维卷积层将创建 ParamAttr 作为参数值。如果未设置初始值，则将偏置初始化为零。默认值：None。


形状：
    - x: :math:`(N, C_{in}, H_{in}, W_{in})`
    - offset: :math:`(N, 2 * H_f * W_f, H_{out}, W_{out})`
    - mask: :math:`(N, H_f * W_f, H_{out}, W_{out})`
    - 输出：:math:`(N, C_{out}, H_{out}, W_{out})`

    其中：

    .. math::

        H_{out} = \frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (kernel\_size[0] - 1) + 1))}{strides[0]} + 1

        W_{out} = \frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (kernel\_size[1] - 1) + 1))}{strides[1]} + 1

代码示例
::::::::::::

COPY-FROM: paddle.vision.ops.DeformConv2D
