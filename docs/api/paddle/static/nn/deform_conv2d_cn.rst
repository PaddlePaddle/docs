.. _cn_api_paddle_static_nn_common_deform_conv2d:

deform_conv2d
-------------------------------


.. py:function:: paddle.static.nn.deform_conv2d(x, offset, mask, num_filters, filter_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, im2col_step=1, weight_attr=None, bias_attr=None, name=None)


**可变形卷积算子**

deform_conv2d op 对输入 4-D Tensor 计算 2-D 可变形卷积。给定输入 Tensor x，输出 Tensor y，可变形卷积运算如下所示：

可形变卷积 v2:

  :math:`y(p) = \sum_{k=1}^{K}{w_k * x(p + p_k + \Delta p_k) * \Delta m_k}`

可形变卷积 v1:

  :math:`y(p) = \sum_{k=1}^{K}{w_k * x(p + p_k + \Delta p_k)}`

其中 :math:`\Delta p_k` 和 :math:`\Delta m_k` 分别为第 k 个位置的可学习偏移和调制标量。在 deform_conv2d_v1 中 :math:`\Delta m_k` 为 1。

具体细节可以参考论文：`Deformable ConvNets v2: More Deformable, Better Results <https://arxiv.org/abs/1811.11168v2>`_ 和 `Deformable Convolutional Networks <https://arxiv.org/abs/1703.06211>`_ 。

**示例**

输入：

    input 形状：:math:`(N, C_{in}, H_{in}, W_{in})`

    卷积核形状：:math:`(C_{out}, C_{in}, H_f, W_f)`

    offset 形状：:math:`(N, 2 * deformable\_groups * H_f * H_w, H_{in}, W_{in})`

    mask 形状：:math:`(N, deformable\_groups * H_f * H_w, H_{in}, W_{in})`

输出：

    输出形状：:math:`(N, C_{out}, H_{out}, W_{out})`

其中

.. math::

    H_{out}&= \frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (H_f - 1) + 1))}{strides[0]} + 1

    W_{out}&= \frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (W_f - 1) + 1))}{strides[1]} + 1


参数
::::::::::::

    - **x** (Tensor) - 形状为 :math:`[N, C, H, W]` 的输入 Tensor，数据类型为 float32 或 float64。
    - **offset** (Tensor) – 可变形卷积层的输入坐标偏移，数据类型为 float32 或 float64。
    - **mask** (Tensor) – 可变形卷积层的输入掩码，当使用可变形卷积算子 v1 时，请将 mask 设置为 None，数据类型为 float32 或 float64。
    - **num_filters** (int) – 卷积核数，与输出 Tensor 通道数相同。
    - **filter_size** (int|tuple) – 卷积核大小。如果 filter_size 为元组，则必须包含两个整数(filter_size_H, filter_size_W)。若数据类型为 int，卷积核形状为(filter_size, filter_size)。
    - **stride** (int|tuple，可选) – 步长大小。如果 stride 为元组，则必须包含两个整数(stride_H, stride_W)。否则 stride_H = stride_W = stride。默认值为 1。
    - **padding** (int|tuple，可选) – padding 大小。如果 padding 为元组，则必须包含两个整数(padding_H, padding_W)。否则 padding_H = padding_W = padding。默认值为 0。
    - **dilation** (int|tuple，可选) – dilation 大小。如果 dilation 为元组，则必须包含两个整数(dilation_H, dilation_W)。否则 dilation_H = dilation_W = dilation。默认值为 1。
    - **groups** (int，可选) – 卷积组数。依据 Alex Krizhevsky 的 Deep CNN 论文中的分组卷积，有：当 group=2 时，前一半卷积核只和前一半输入通道有关，而后一半卷积核只和后一半输入通道有关。默认值为 1。
    - **deformable_groups** (int，可选) – 可变形卷积组数。默认值为 1。
    - **im2col_step** (int，可选) – 每个 im2col 计算的最大图像数。总 batch 大小应可以被该值整除或小于该值。如果您面临内存问题，可以尝试在此处使用一个较小的值。默认值为 1。
    - **weight_attr** (ParamAttr，可选) – 可变形卷积的可学习权重的属性。如果将其设置为 None 或某种 ParamAttr，可变形卷积将创建 ParamAttr 作为 weight_attr。如果没有设置此 weight_attr 的 Initializer，该参数将被 Normal(0.0, std)初始化，且其中的 std 为 :math:`(\frac{2.0 }{filter\_elem\_num})^{0.5}`。默认值为 None。
    - **bias_attr** (ParamAttr|bool，可选) – 可变形卷积层的偏置的参数属性。如果设为 False，则输出单元不会加偏置。如果设为 None 或者某种 ParamAttr，conv2d 会创建 ParamAttr 作为 bias_attr。如果不设置 bias_attr 的 Initializer，偏置会被初始化为 0。默认值为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，可变形卷积输出的 4-D Tensor，数据类型为 float32 或 float64。

代码示例
::::::::::::

COPY-FROM: paddle.static.nn.deform_conv2d
