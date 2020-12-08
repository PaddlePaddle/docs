DeformConv2D
-------------------------------

.. py:class:: paddle.vision.ops.DeformConv2D(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, weight_attr=None, bias_attr=None)


deform_conv2d 对输入4-D Tensor计算2-D可变形卷积。给定输入Tensor x，输出Tensor y，可变形卷积运算如下所示：

可形变卷积v2(make != None):

  :math:`y(p) = \sum_{k=1}^{K}{w_k * x(p + p_k + \Delta p_k) * \Delta m_k}`

可形变卷积v1(make = None):

  :math:`y(p) = \sum_{k=1}^{K}{w_k * x(p + p_k + \Delta p_k)}`

其中 :math:`\Delta p_k` 和 :math:`\Delta m_k` 分别为第k个位置的可学习偏移和调制标量。在deformable conv v1中 :math:`\Delta m_k` 为1.

具体细节可以参考论文：`<<Deformable ConvNets v2: More Deformable, Better Results>> <https://arxiv.org/abs/1811.11168v2>`_ 和 `<<Deformable Convolutional Networks>> <https://arxiv.org/abs/1703.06211>`_ 。

**示例**
     
输入：
    input 形状： :math:`(N, C_{in}, H_{in}, W_{in})`

    卷积核形状： :math:`(C_{out}, C_{in}, H_f, W_f)`

    offset 形状： :math:`(N, 2 * H_f * W_f, H_{out}, W_{out})`

    mask 形状： :math:`(N, H_f * W_f, H_{out}, W_{out})`
     
输出：
    输出形状： :math:`(N, C_{out}, H_{out}, W_{out})`

其中

.. math::

    H_{out}&= \frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (H_f - 1) + 1))}{strides[0]} + 1

    W_{out}&= \frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (W_f - 1) + 1))}{strides[1]} + 1


参数：
    - **in_channels** (int) - 输入图像的通道数。
    - **out_channels** (int) - 由卷积操作产生的输出的通道数。
    - **kernel_size** (int|list|tuple) - 卷积核大小。可以为单个整数或包含两个整数的元组或列表，分别表示卷积核的高和宽。如果为单个整数，表示卷积核的高和宽都等于该整数。
    - **stride** (int|list|tuple，可选) - 步长大小。可以为单个整数或包含两个整数的元组或列表，分别表示卷积沿着高和宽的步长。如果为单个整数，表示沿着高和宽的步长都等于该整数。默认值：1。
    - **padding** (int|list|tuple，可选) - 填充大小。卷积核操作填充大小。如果它是一个列表或元组，则必须包含两个整型数：（padding_height,padding_width）。若为一个整数，padding_height = padding_width = padding。默认值：0。
    - **dilation** (int|list|tuple，可选) - 空洞大小。可以为单个整数或包含两个整数的元组或列表，分别表示卷积核中的元素沿着高和宽的空洞。如果为单个整数，表示高和宽的空洞都等于该整数。默认值：1。
    - **groups** (int，可选) - 二维卷积层的组数。根据Alex Krizhevsky的深度卷积神经网络（CNN）论文中的成组卷积：当group=n，输入和卷积核分别根据通道数量平均分为n组，第一组卷积核和第一组输入进行卷积计算，第二组卷积核和第二组输入进行卷积计算，……，第n组卷积核和第n组输入进行卷积计算。默认值：1。
    - **weight_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **bias_attr** （ParamAttr|bool，可选）- 指定偏置参数属性的对象。若 ``bias_attr`` 为bool类型，只支持为False，表示没有偏置参数。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。

    
形状:
    - x: :math:`(N, C_{in}, H_{in}, W_{in})`
    - offset: :math:`(N, 2 * H_f * W_f, H_{out}, W_{out})`
    - mask: :math:`(N, H_f * W_f, H_{out}, W_{out})`
    - 输出: :math:`(N, C_{out}, H_{out}, W_{out})`

    其中:

    .. math::

        H_{out} = \frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (kernel\_size[0] - 1) + 1))}{strides[0]} + 1

        W_{out} = \frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (kernel\_size[1] - 1) + 1))}{strides[1]} + 1

**代码示例**：

.. code-block:: python

   #deformable conv v2:

   import paddle
   input = paddle.rand((8, 1, 28, 28))
   kh, kw = 3, 3
   # offset shape should be [bs, 2 * kh * kw, out_h, out_w]
   # mask shape should be [bs, hw * hw, out_h, out_w]
   # In this case, for an input of 28, stride of 1
   # and kernel size of 3, without padding, the output size is 26
   offset = paddle.rand((8, 2 * kh * kw, 26, 26))
   mask = paddle.rand((8, kh * kw, 26, 26))
   deform_conv = paddle.vision.ops.DeformConv2D(
       in_channels=1,
       out_channels=16,
       kernel_size=[kh, kw])
   out = deform_conv(input, offset, mask)
   print(out.shape)
   # returns
   [8, 16, 26, 26]

   #deformable conv v1:

   import paddle
   input = paddle.rand((8, 1, 28, 28))
   kh, kw = 3, 3
   # offset shape should be [bs, 2 * kh * kw, out_h, out_w]
   # mask shape should be [bs, hw * hw, out_h, out_w]
   # In this case, for an input of 28, stride of 1
   # and kernel size of 3, without padding, the output size is 26
   offset = paddle.rand((8, 2 * kh * kw, 26, 26))
   deform_conv = paddle.vision.ops.DeformConv2D(
       in_channels=1,
       out_channels=16,
       kernel_size=[kh, kw])
   out = deform_conv(input, offset)
   print(out.shape)
   # returns
   [8, 16, 26, 26]
