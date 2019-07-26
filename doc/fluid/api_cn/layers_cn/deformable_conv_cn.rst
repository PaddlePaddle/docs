.. _cn_api_fluid_layers_deformable_conv:

deformable_conv
-------------------------------

.. py:function:: paddle.fluid.layers.deformable_conv(input, offset, mask, num_filters, filter_size, stride=1, padding=0, dilation=1, groups=None, deformable_groups=None, im2col_step=None, param_attr=None, bias_attr=None, name=None)

可变形卷积层

在4-D输入上计算2-D可变形卷积。给定输入图像x，输出特征图y，可变形卷积操作如下所示：
\[y(p) = \sum_{k=1}^{K}{w_k * x(p + p_k + \Delta p_k) * \Delta m_k}\]
其中\(\Delta p_k\) 和 \(\Delta m_k\) 分别为第k个位置的可学习偏移和调制标量。
参考可变形卷积网络v2：可变形程度越高，结果越好。

**示例**
     
输入：
    输入形状： \((N, C_{in}, H_{in}, W_{in})\)
    卷积核形状： \((C_{out}, C_{in}, H_f, W_f)\)
    偏移形状： \((N, 2 * deformable\_groups * H_f * H_w, H_{in}, W_{in})\)
    掩膜形状： \((N, deformable\_groups * H_f * H_w, H_{in}, W_{in})\)
     
输出：
    输出形状： \((N, C_{out}, H_{out}, W_{out})\)

其中
    \[\begin{split}H_{out}&= \frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (H_f - 1) + 1))}{strides[0]} + 1 \\
    W_{out}&= \frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (W_f - 1) + 1))}{strides[1]} + 1\end{split}\]
     

参数：
    - **input** (Variable) - 形式为[N, C, H, W]的输入图像。
    - **offset** (Variable) – 可变形卷积层的输入坐标偏移。
    - **Mask** (Variable) – 可变形卷积层的输入掩膜。
    - **num_filters** (int) – 卷积核数。和输出图像通道数相同。
    - **filter_size** (int|tuple|None) – 卷积核大小。如果filter_size为元组，则必须包含两个整数(filter_size_H, filter_size_W)。否则卷积核将为方形。
    - **stride** (int|tuple) – 步长大小。如果stride为元组，则必须包含两个整数(stride_H, stride_W)。否则stride_H = stride_W = stride。默认stride = 1。
    - **padding** (int|tuple) – padding大小。如果padding为元组，则必须包含两个整数(padding_H, padding_W)。否则padding_H = padding_W = padding。默认padding = 0。
    - **dilation** (int|tuple) – dilation大小。如果dilation为元组，则必须包含两个整数(dilation_H, dilation_W)。否则dilation_H = dilation_W = dilation。默认dilation = 1。
    - **groups** (int) – 可变形卷积层的群组数。依据Alex Krizhevsky的Deep CNN论文中的分组卷积，有：当group=2时，前一半卷积核只和前一半输入通道有关，而后一半卷积核只和后一半输入通道有关。默认groups=1。
    - **deformable_groups** (int) – 可变形群组分区数。默认deformable_groups = 1。
    - **im2col_step** (int) – 每个im2col计算的最大图像数。总batch大小应该可以被该值整除或小于该值。如果您面临内存问题，可以尝试在此处使用一个更小的值。默认im2col_step = 64。
    - **param_attr** (ParamAttr|None) – 可变形卷积的可学习参数/权重的参数属性。如果将其设置为None或ParamAttr的一个属性，可变形卷积将创建ParamAttr作为param_attr。如果没有设置此param_attr的Initializer，该参数将被\(Normal(0.0, std)\)初始化，且其中的\(std\) 为 \((\frac{2.0 }{filter\_elem\_num})^{0.5}\)。默认值None。
    - **bias_attr** (ParamAttr|bool|None) – 可变形卷积层的偏置的参数属性。如果设为False，则输出单元不会加偏置。如果设为None或者ParamAttr的一个属性，conv2d会创建ParamAttr作为bias_attr。如果不设置bias_attr的Initializer，偏置会被初始化为0。默认值None。
    - **name** (str|None) – 该层的名字（可选项）。如果设为None，该层将会被自动命名。默认值None。
 
返回：储存可变形卷积结果的张量变量。
     
返回类型：变量(Variable)
     
抛出：ValueError – 如果input, filter_size, stride, padding和groups的大小不匹配。

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(name='data', shape=[3, 32, 32], dtype='float32')
    offset = fluid.layers.data(name='offset', shape=[18, 32, 32], dtype='float32')
    mask = fluid.layers.data(name='mask', shape=[9, 32, 32], dtype='float32')
    out = fluid.layers.deformable_conv(input=data, offset=offset, mask=mask, num_filters=2, filter_size=3, padding=1)






