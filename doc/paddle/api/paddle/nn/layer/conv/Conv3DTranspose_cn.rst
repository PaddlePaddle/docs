Conv3DTranspose
-------------------------------

.. py:class:: paddle.nn.Conv3DTranspose(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, dilation=1, weight_attr=None, bias_attr=None, data_format="NCDHW")


三维转置卷积层（Convlution3d transpose layer)

该层根据输入（input）、卷积核（kernel）和卷积核空洞大小（dilations）、步长（stride）、填充（padding）来计算输出特征层大小或者通过output_size指定输出特征层大小。输入(Input)和输出(Output)为NCDHW或者NDHWC格式。其中N为批尺寸，C为通道数（channel），D为特征深度，H为特征层高度，W为特征层宽度。转置卷积的计算过程相当于卷积的反向计算。转置卷积又被称为反卷积（但其实并不是真正的反卷积）。欲了解卷积转置层细节，请参考下面的说明和 参考文献_ 。如果参数bias_attr不为False, 转置卷积计算会添加偏置项。

.. _参考文献: https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf

输入 :math:`X` 和输出 :math:`Out` 函数关系如下：

.. math::
                        \\Out=\sigma (W*X+b)\\

其中：
    -  :math:`X` : 输入，具有NCDHW或NDHWC格式的5-D Tensor
    -  :math:`W` : 卷积核，具有NCDHW格式的5-D Tensor
    -  :math:`*` : 卷积操作（注意：转置卷积本质上的计算还是卷积）
    -  :math:`b` : 偏置（bias），2-D Tensor，形状为 ``[M,1]``
    -  :math:`σ` : 激活函数
    -  :math:`Out` : 输出值，NCDHW或NDHWC格式的5-D Tensor，和 ``X`` 的形状可能不同


注意：

如果output_size为None，则 :math:`H_{out}` = :math:`H^\prime_{out}` , :math:`W_{out}` = :math:`W^\prime_{out}` ;否则，指定的output_size_height（输出特征层的高） :math:`H_{out}` 应当介于 :math:`H^\prime_{out}` 和 :math:`H^\prime_{out} + strides[0]` 之间（不包含 :math:`H^\prime_{out} + strides[0]` ）, 并且指定的output_size_width（输出特征层的宽） :math:`W_{out}` 应当介于 :math:`W^\prime_{out}` 和 :math:`W^\prime_{out} + strides[1]` 之间（不包含 :math:`W^\prime_{out} + strides[1]` ）。

由于转置卷积可以当成是卷积的反向计算，而根据卷积的输入输出计算公式来说，不同大小的输入特征层可能对应着相同大小的输出特征层，所以对应到转置卷积来说，固定大小的输入特征层对应的输出特征层大小并不唯一。

如果指定了output_size， 该算子可以自动计算卷积核的大小。

参数:
  - **in_channels** (int) - 输入图像的通道数。
  - **out_channels** (int) - 卷积核的个数，和输出特征图个数相同。
  - **kernel_size** (int|list|tuple) - 卷积核大小。可以为单个整数或包含三个整数的元组或列表，分别表示卷积核的深度，高和宽。如果为单个整数，表示卷积核的深度，高和宽都等于该整数。默认：None。output_size和kernel_size不能同时为None。
  - **stride** (int|tuple, 可选) - 步长大小。如果 ``stride`` 为元组或列表，则必须包含三个整型数，分别表示深度，垂直和水平滑动步长。否则，表示深度，垂直和水平滑动步长均为 ``stride`` 。默认值：1。
  - **padding** (int|tuple, 可选) - 填充大小。如果 ``padding`` 为元组或列表，则必须包含三个整型数，分别表示深度，竖直和水平边界填充大小。否则，表示深度，竖直和水平边界填充大小均为 ``padding`` 。如果它是一个字符串，可以是"VALID"或者"SAME"，表示填充算法，计算细节可参考下方形状 ``padding`` = "SAME"或  ``padding`` = "VALID" 时的计算公式。默认值：0。
  - **output_padding** (int|list|tuple, optional): 输出形状上一侧额外添加的大小. 默认值: 0.
  - **groups** (int, 可选) - 二维卷积层的组数。根据Alex Krizhevsky的深度卷积神经网络（CNN）论文中的分组卷积：当group=2，卷积核的前一半仅和输入特征图的前一半连接。卷积核的后一半仅和输入特征图的后一半连接。默认值：1。
  - **dilation** (int|tuple, 可选) - 空洞大小。可以为单个整数或包含三个整数的元组或列表，分别表示卷积核中的元素沿着深度，高和宽的空洞。如果为单个整数，表示深度，高和宽的空洞都等于该整数。默认值：1。
  - **weight_attr** (ParamAttr, 可选) - 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
  - **bias_attr** (ParamAttr|bool, 可选) - 指定偏置参数属性的对象。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
  - **data_format** (str，可选) - 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCHW"和"NHWC"。N是批尺寸，C是通道数，H是特征高度，W是特征宽度。默认值："NCDHW"。

形状:

    - 输入：:math:`（N，C_{in}， H_{in}， W_{in}）`


    - 输出：:math:`（N，C_{out}, H_{out}, W_{out}）`

    其中

    .. math::

        & D'_{out}=(D_{in}-1)*strides[0] - pad\_depth\_front - pad\_depth\_back + dilations[0]*(kernel\_size[0]-1)+1\\
        & H'_{out} = (H_{in}-1)*strides[1] - pad\_height\_top - pad\_height\_bottom + dilations[1]*(kernel\_size[1]-1)+1\\
        & W'_{out} = (W_{in}-1)*strides[2]- pad\_width\_left - pad\_width\_right + dilations[2]*(kernel\_size[2]-1)+1 \\
        & D_{out}\in[D'_{out},D'_{out} + strides[0])\\
        & H_{out}\in[H'_{out},H'_{out} + strides[1])\\
        & W_{out}\in[W'_{out},W'_{out} + strides[2])\\

    如果 ``padding`` = "SAME":

    .. math::
        & D'_{out} = \frac{(D_{in} + stride[0] - 1)}{stride[0]}\\
        & H'_{out} = \frac{(H_{in} + stride[1] - 1)}{stride[1]}\\
        & W'_{out} = \frac{(W_{in} + stride[2] - 1)}{stride[2]}\\

    如果 ``padding`` = "VALID":

    .. math::
        & D'_{out} = (D_{in}-1)*strides[0] + dilations[0]*(kernel\_size[0]-1)+1\\
        & H'_{out} = (H_{in}-1)*strides[1] + dilations[1]*(kernel\_size[1]-1)+1\\
        & W'_{out} = (W_{in}-1)*strides[2] + dilations[2]*(kernel\_size[2]-1)+1 \\

抛出异常:
    -  ``ValueError`` : 如果输入的shape、kernel_size、stride、padding和groups不匹配，抛出ValueError
    -  ``ValueError`` - 如果 ``data_format`` 既不是"NCHW"也不是"NHWC"。
    -  ``ValueError`` - 如果 ``padding`` 是字符串，既不是"SAME"也不是"VALID"。
    -  ``ValueError`` - 如果 ``padding`` 含有4个二元组，与批尺寸对应维度的值不为0或者与通道对应维度的值不为0。
    -  ``ValueError`` - 如果 ``output_size`` 和 ``filter_size`` 同时为None。
    -  ``ShapeError`` - 如果输入不是4-D Tensor。
    -  ``ShapeError`` - 如果输入和卷积核的维度大小不相同。
    -  ``ShapeError`` - 如果输入的维度大小与 ``stride`` 之差不是2。

**代码示例**

..  code-block:: python

    import paddle
    import paddle.nn as nn

    x_var = paddle.uniform((2, 4, 8, 8, 8), dtype='float32', min=-1., max=1.)

    conv = nn.Conv3DTranspose(4, 6, (3, 3, 3))
    y_var = conv(x_var)
    y_np = y_var.numpy()
    print(y_np.shape)
    # (2, 6, 10, 10, 10)

