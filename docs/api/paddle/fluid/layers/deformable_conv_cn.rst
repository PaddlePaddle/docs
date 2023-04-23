.. _cn_api_fluid_layers_deformable_conv:

deformable_conv
-------------------------------


.. py:function:: paddle.fluid.layers.deformable_conv(input, offset, mask, num_filters, filter_size, stride=1, padding=0, dilation=1, groups=None, deformable_groups=None, im2col_step=None, param_attr=None, bias_attr=None, modulated=True, name=None)




**可变形卷积算子**

deformable_conv op对输入4-D Tensor计算2-D可变形卷积。给定输入Tensor x，输出Tensor y，可变形卷积运算如下所示：

可形变卷积v2:

  :math:`y(p) = \sum_{k=1}^{K}{w_k * x(p + p_k + \Delta p_k) * \Delta m_k}`

可形变卷积v1:

  :math:`y(p) = \sum_{k=1}^{K}{w_k * x(p + p_k + \Delta p_k)}`

其中 :math:`\Delta p_k` 和 :math:`\Delta m_k` 分别为第k个位置的可学习偏移和调制标量。在deformable_conv_v1中 :math:`\Delta m_k` 为1。

具体细节可以参考论文：`<<Deformable ConvNets v2: More Deformable, Better Results>> <https://arxiv.org/abs/1811.11168v2>`_ 和 `<<Deformable Convolutional Networks>> <https://arxiv.org/abs/1703.06211>`_ 。

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

    - **input** (Variable) - 形状为 :math:`[N, C, H, W]` 的输入Tensor，数据类型为float32或float64。
    - **offset** (Variable) – 可变形卷积层的输入坐标偏移，数据类型为float32或float64。
    - **mask** (Variable，可选) – 可变形卷积层的输入掩码，当使用可变形卷积算子v1时，请将mask设置为None，数据类型为float32或float64。
    - **num_filters** (int) – 卷积核数，与输出Tensor通道数相同。
    - **filter_size** (int|tuple) – 卷积核大小。如果filter_size为元组，则必须包含两个整数(filter_size_H, filter_size_W)。若数据类型为int，卷积核形状为(filter_size, filter_size)。
    - **stride** (int|tuple) – 步长大小。如果stride为元组，则必须包含两个整数(stride_H, stride_W)。否则stride_H = stride_W = stride。默认值为1。
    - **padding** (int|tuple) – padding大小。如果padding为元组，则必须包含两个整数(padding_H, padding_W)。否则padding_H = padding_W = padding。默认值为0。
    - **dilation** (int|tuple) – dilation大小。如果dilation为元组，则必须包含两个整数(dilation_H, dilation_W)。否则dilation_H = dilation_W = dilation。默认值为1。
    - **groups** (int) – 卷积组数。依据Alex Krizhevsky的Deep CNN论文中的分组卷积，有：当group=2时，前一半卷积核只和前一半输入通道有关，而后一半卷积核只和后一半输入通道有关。缺省值为1。
    - **deformable_groups** (int) – 可变形卷积组数。默认值为1。
    - **im2col_step** (int) – 每个im2col计算的最大图像数。总batch大小应可以被该值整除或小于该值。如果您面临内存问题，可以尝试在此处使用一个较小的值。默认值为64。
    - **param_attr** (ParamAttr，可选) – 可变形卷积的可学习权重的属性。如果将其设置为None或某种ParamAttr，可变形卷积将创建ParamAttr作为param_attr。如果没有设置此param_attr的Initializer，该参数将被Normal(0.0, std)初始化，且其中的std为 :math:`(\frac{2.0 }{filter\_elem\_num})^{0.5}`。默认值为None。
    - **bias_attr** (ParamAttr|bool，可选) – 可变形卷积层的偏置的参数属性。如果设为False，则输出单元不会加偏置。如果设为None或者某种ParamAttr，conv2d会创建ParamAttr作为bias_attr。如果不设置bias_attr的Initializer，偏置会被初始化为0。默认值为None。
    - **modulated** （bool）- 确定使用v1和v2中的哪个版本，如果为True，则选择使用v2。默认值为True。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
 
返回
::::::::::::
可变形卷积输出的4-D Tensor，数据类型为float32或float64。
     
返回类型
::::::::::::
Variable
     
抛出异常
::::::::::::
ValueError – 如果input, filter_size, stride, padding和groups的大小不匹配。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.deformable_conv