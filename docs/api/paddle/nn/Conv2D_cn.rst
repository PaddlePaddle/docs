.. _cn_api_paddle_nn_Conv2D:

Conv2D
-------------------------------

.. py:class:: paddle.nn.Conv2D(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', weight_attr=None, bias_attr=None, data_format="NCHW")



**二维卷积层**

根据输入、卷积核、步长（stride）、填充（padding）、空洞大小（dilations）一组参数计算输出特征层大小。输入和输出是 NCHW 或 NHWC 格式，其中 N 是批尺寸，C 是通道数，H 是特征高度，W 是特征宽度。卷积核是 MCHW 格式，M 是输出图像通道数，C 是输入图像通道数，H 是卷积核高度，W 是卷积核宽度。如果组数(groups)大于 1，C 等于输入图像通道数除以组数的结果。详情请参考 UFLDL's : `卷积 <http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/>`_ 。如果 bias_attr 不为 False，卷积计算会添加偏置项。

对每个输入 X，有等式：

.. math::

    Out = \sigma \left ( W * X + b \right )

其中：

    - :math:`X`：输入值，NCHW 或 NHWC 格式的 4-D Tensor
    - :math:`W`：卷积核值，MCHW 格式的 4-D Tensor
    - :math:`*`：卷积操作
    - :math:`b`：偏置值，1-D Tensor，形状为 ``[M]``
    - :math:`\sigma`：激活函数
    - :math:`Out`：输出值，NCHW 或 NHWC 格式的 4-D Tensor，和 ``X`` 的形状可能不同


参数
::::::::::::

    - **in_channels** (int) - 输入图像的通道数。
    - **out_channels** (int) - 由卷积操作产生的输出的通道数。
    - **kernel_size** (int|list|tuple) - 卷积核大小。可以为单个整数或包含两个整数的元组或列表，分别表示卷积核的高和宽。如果为单个整数，表示卷积核的高和宽都等于该整数。
    - **stride** (int|list|tuple，可选) - 步长大小。可以为单个整数或包含两个整数的元组或列表，分别表示卷积沿着高和宽的步长。如果为单个整数，表示沿着高和宽的步长都等于该整数。默认值：1。
    - **padding** (int|list|tuple|str，可选) - 填充大小。如果它是一个字符串，可以是"VALID"或者"SAME"，表示填充算法，计算细节可参考上述 ``padding`` = "SAME"或  ``padding`` = "VALID" 时的计算公式。如果它是一个元组或列表，它可以有 3 种格式：(1)包含 4 个二元组：当 ``data_format`` 为"NCHW"时为 [[0,0], [0,0], [padding_height_top, padding_height_bottom], [padding_width_left, padding_width_right]]，当 ``data_format`` 为"NHWC"时为[[0,0], [padding_height_top, padding_height_bottom], [padding_width_left, padding_width_right], [0,0]]；(2)包含 4 个整数值：[padding_height_top, padding_height_bottom, padding_width_left, padding_width_right]；(3)包含 2 个整数值：[padding_height, padding_width]，此时 padding_height_top = padding_height_bottom = padding_height， padding_width_left = padding_width_right = padding_width。若为一个整数，padding_height = padding_width = padding。默认值：0。
    - **dilation** (int|list|tuple，可选) - 空洞大小。可以为单个整数或包含两个整数的元组或列表，分别表示卷积核中的元素沿着高和宽的空洞。如果为单个整数，表示高和宽的空洞都等于该整数。默认值：1。
    - **groups** (int，可选) - 二维卷积层的组数。根据 Alex Krizhevsky 的深度卷积神经网络（CNN）论文中的成组卷积：当 group=n，输入和卷积核分别根据通道数量平均分为 n 组，第一组卷积核和第一组输入进行卷积计算，第二组卷积核和第二组输入进行卷积计算，……，第 n 组卷积核和第 n 组输入进行卷积计算。默认值：1。
    - **padding_mode** (str，可选)：填充模式。包括 ``'zeros'``, ``'reflect'``, ``'replicate'`` 或者 ``'circular'``。默认值：``'zeros'`` 。
    - **weight_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值为 None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **bias_attr** （ParamAttr|bool，可选）- 指定偏置参数属性的对象。若 ``bias_attr`` 为 bool 类型，只支持为 False，表示没有偏置参数。默认值为 None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **data_format** (str，可选) - 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCHW"和"NHWC"。N 是批尺寸，C 是通道数，H 是特征高度，W 是特征宽度。默认值："NCHW"。


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
    - 输入：:math:`(N, C_{in}, H_{in}, W_{in})`

    - 卷积核：:math:`(C_{out}, C_{in}, K_{h}, K_{w})`

    - 偏置：:math:`(C_{out})`

    - 输出：:math:`(N, C_{out}, H_{out}, W_{out})`

    其中：

    .. math::
        H_{out} = \frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (kernel\_size[0] - 1) + 1))}{strides[0]} + 1

        W_{out} = \frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (kernel\_size[1] - 1) + 1))}{strides[1]} + 1

    如果 ``padding`` = "SAME":

    .. math::
        H_{out} = \frac{(H_{in} + stride[0] - 1)}{stride[0]}

    .. math::
        W_{out} = \frac{(W_{in} + stride[1] - 1)}{stride[1]}

    如果 ``padding`` = "VALID":

    .. math::
        H_{out} = \frac{\left ( H_{in} -\left ( dilation[0]*\left ( kernel\_size[0]-1 \right )+1 \right ) \right )}{stride[0]}+1

        W_{out} = \frac{\left ( W_{in} -\left ( dilation[1]*\left ( kernel\_size[1]-1 \right )+1 \right ) \right )}{stride[1]}+1



代码示例
::::::::::::
COPY-FROM: paddle.nn.Conv2D
