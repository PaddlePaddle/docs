.. _cn_api_paddle_vision_ops_deform_conv2d:

deform_conv2d
-------------------------------

.. py:function:: paddle.vision.ops.deform_conv2d(x, offset, weight, bias=None, stride=1, padding=0, dilation=1, deformable_groups=1, groups=1, mask=None, name=None)

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

    - **x** (Tensor) - 形状为 :math:`[N, C, H, W]` 的输入 Tensor，数据类型为 float32 或 float64。
    - **offset** (Tensor) – 可变形卷积层的输入坐标偏移，数据类型为 float32 或 float64。
    - **weight** (Tensor) – 卷积核参数，形状为 :math:`[M, C/g, kH, kW]`，其中 M 是输出通道数，g 是 group 组数，kH 是卷积核高度尺寸，kW 是卷积核宽度尺寸。数据类型为 float32 或 float64。
    - **bias** (Tensor，可选) - 可变形卷积偏置项，形状为 :math:`[M,]` 。
    - **stride** (int|list|tuple，可选) - 步长大小。卷积核和输入进行卷积计算时滑动的步长。如果它是一个列表或元组，则必须包含两个整型数：（stride_height,stride_width）。若为一个整数，stride_height = stride_width = stride。默认值：1。
    - **padding** (int|list|tuple，可选) - 填充大小。卷积核操作填充大小。如果它是一个列表或元组，则必须包含两个整型数：（padding_height,padding_width）。若为一个整数，padding_height = padding_width = padding。默认值：0。
    - **dilation** (int|list|tuple，可选) - 空洞大小。空洞卷积时会使用该参数，卷积核对输入进行卷积时，感受野里每相邻两个特征点之间的空洞信息。如果空洞大小为列表或元组，则必须包含两个整型数：（dilation_height,dilation_width）。若为一个整数，dilation_height = dilation_width = dilation。默认值：1。
    - **deformable_groups** (int，可选) - 可变形卷积组数。默认值：1。
    - **groups** (int，可选) - 二维卷积层的组数。根据 Alex Krizhevsky 的深度卷积神经网络（CNN）论文中的成组卷积：当 group=n，输入和卷积核分别根据通道数量平均分为 n 组，第一组卷积核和第一组输入进行卷积计算，第二组卷积核和第二组输入进行卷积计算，……，第 n 组卷积核和第 n 组输入进行卷积计算。默认值：1。
    - **mask** (Tensor，可选) – 可变形卷积层的输入掩码，当使用可变形卷积算子 v1 时，请将 mask 设置为 None，数据类型为 float32 或 float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
可变形卷积输出的 4-D Tensor，数据类型为 float32 或 float64。


代码示例
::::::::::::

COPY-FROM: paddle.vision.ops.deform_conv2d
