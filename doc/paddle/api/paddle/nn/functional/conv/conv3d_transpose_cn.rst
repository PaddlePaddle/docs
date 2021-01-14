conv3d_transpose
-------------------------------


.. py:function:: paddle.nn.functional.conv3d_transpose(x, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1, data_format='NCHW', output_size=None, name=None)




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

**示例**

输入:

    输入的shape：:math:`（N,C_{in}, D_{in}, H_{in}, W_{in}）`

    卷积核的shape：:math:`（C_{in}, C_{out}, D_f, H_f, W_f）`

输出:

    输出的shape：:math:`（N,C_{out}, D_{out}, H_{out}, W_{out}）`


其中：

.. math::

    & D'_{out}=(D_{in}-1)*strides[0] - pad\_depth\_front - pad\_depth\_back + dilations[0]*(D_f-1)+1\\
    & H'_{out}=(H_{in}-1)*strides[1] - pad\_height\_top - pad\_height\_bottom + dilations[1]*(H_f-1)+1\\
    & W'_{out}=(W_{in}-1)*strides[2] - pad\_width\_left - pad\_width\_right + dilations[2]*(W_f-1)+1\\
    & D_{out}\in[D'_{out},D'_{out} + strides[0])\\
    & H_{out}\in[H'_{out},H'_{out} + strides[1])\\
    & W_{out}\in[W'_{out},W'_{out} + strides[2])\\

如果 ``padding`` = "SAME":

.. math::
    D'_{out} = \frac{(D_{in} + stride[0] - 1)}{stride[0]}\\
    H'_{out} = \frac{(H_{in} + stride[1] - 1)}{stride[1]}\\
    W'_{out} = \frac{(W_{in} + stride[2] - 1)}{stride[2]}\\

如果 ``padding`` = "VALID":

.. math::
    D'_{out}=(D_{in}-1)*strides[0] + dilations[0]*(D_f-1)+1\\
    H'_{out}=(H_{in}-1)*strides[1] + dilations[1]*(H_f-1)+1\\
    W'_{out}=(W_{in}-1)*strides[2] + dilations[2]*(W_f-1)+1\\

注意：

如果output_size为None，则 :math:`D_{out}` = :math:`D^\prime_{out}` , :math:`H_{out}` = :math:`H^\prime_{out}` , :math:`W_{out}` = :math:`W^\prime_{out}` ;否则，指定的output_size_depth（输出特征层的深度） :math:`D_{out}` 应当介于 :math:`D^\prime_{out}` 和 :math:`D^\prime_{out} + strides[0]` 之间（不包含 :math:`D^\prime_{out} + strides[0]` ），指定的output_size_height（输出特征层的高） :math:`H_{out}` 应当介于 :math:`H^\prime_{out}` 和 :math:`H^\prime_{out} + strides[1]` 之间（不包含 :math:`H^\prime_{out} + strides[1]` ）, 并且指定的output_size_width（输出特征层的宽） :math:`W_{out}` 应当介于 :math:`W^\prime_{out}` 和 :math:`W^\prime_{out} + strides[2]` 之间（不包含 :math:`W^\prime_{out} + strides[2]` ）。

由于转置卷积可以当成是卷积的反向计算，而根据卷积的输入输出计算公式来说，不同大小的输入特征层可能对应着相同大小的输出特征层，所以对应到转置卷积来说，固定大小的输入特征层对应的输出特征层大小并不唯一。

如果指定了output_size， ``conv3d_transpose`` 可以自动计算卷积核的大小。

参数:
  - **x** (Tensor) - 形状为 :math:`[N, C, D, H, W]` 或 :math:`[N, D, H, W, C]` 的5-D Tensor，N是批尺寸，C是通道数，D是特征深度，H是特征高度，W是特征宽度，数据类型：float32或float64。
  - **weight** (Tensor) - 形状为 :math:`[C, M/g, kD, kH, kW]` 的卷积核。 M是输出通道数， g是分组的个数，kD是卷积核的深度，kH是卷积核的高度，kW是卷积核的宽度。
  - **bias** (int|list|tuple) - 偏置项，形状为： :math:`[M,]` 。
  - **stride** (int|list|tuple，可选) - 步长大小。如果 ``stride`` 为元组或列表，则必须包含三个整型数，分别表示深度，垂直和水平滑动步长。否则，表示深度，垂直和水平滑动步长均为 ``stride`` 。默认值：1。
  - **padding** (int|list|tuple|str，可选) - 填充padding大小。padding参数在输入特征层每边添加 ``dilation * (kernel_size - 1) - padding`` 个0。如果它是一个字符串，可以是"VALID"或者"SAME"，表示填充算法，计算细节可参考上述 ``padding`` = "SAME"或  ``padding`` = "VALID" 时的计算公式。如果它是一个元组或列表，它可以有3种格式：(1)包含5个二元组：当 ``data_format`` 为"NCDHW"时为 [[0,0], [0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]，当 ``data_format`` 为"NDHWC"时为[[0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]；(2)包含6个整数值：[pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]；(3)包含3个整数值：[pad_depth, pad_height, pad_width]，此时 pad_depth_front = pad_depth_back = pad_depth, pad_height_top = pad_height_bottom = pad_height, pad_width_left = pad_width_right = pad_width。若为一个整数，pad_depth = pad_height = pad_width = padding。默认值：0。
  - **output_padding** (int|list|tuple, optional): 输出形状上一侧额外添加的大小. 默认值: 0.
  - **dilation** (int|list|tuple，可选) - 空洞大小。空洞卷积时会使用该参数，卷积核对输入进行卷积时，感受野里每相邻两个特征点之间的空洞信息。如果空洞大小为列表或元组，则必须包含两个整型数：（dilation_height,dilation_width）。若为一个整数，dilation_height = dilation_width = dilation。默认值：1。
  - **groups** (int，可选) - 三维转置卷积层的组数。从Alex Krizhevsky的CNN Deep论文中的群卷积中受到启发，当group=2时，输入和卷积核分别根据通道数量平均分为两组，第一组卷积核和第一组输入进行卷积计算，第二组卷积核和第二组输入进行卷积计算。默认：group = 1。
  - **weight_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
  - **bias_attr** （ParamAttr|bool，可选）- 指定偏置参数属性的对象。若 ``bias_attr`` 为bool类型，只支持为False，表示没有偏置参数。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
  - **data_format** (str，可选) - 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCHW"和"NHWC"。N是批尺寸，C是通道数，H是特征高度，W是特征宽度。默认值："NCHW"。
  - **name** (str，可选) – 具体用法请参见 :ref:`cn_api_guide_Name` ，一般无需设置，默认值：None。

  
返回：5-D Tensor，数据类型与 ``input`` 一致。如果未指定激活层，则返回转置卷积计算的结果，如果指定激活层，则返回转置卷积和激活计算之后的最终结果。

返回类型：Tensor

抛出异常:
    - ``ValueError`` - 如果输入的shape、kernel_size、stride、padding和groups不匹配。
    - ``ValueError`` - 如果 ``data_format`` 既不是"NCDHW"也不是"NDHWC"。
    - ``ValueError`` - 如果 ``padding`` 是字符串，既不是"SAME"也不是"VALID"。
    - ``ValueError`` - 如果 ``padding`` 含有5个二元组，与批尺寸对应维度的值不为0或者与通道对应维度的值不为0。
    - ``ValueError`` - 如果 ``output_size`` 和 ``filter_size`` 同时为None。
    - ``ShapeError`` - 如果输入不是5-D Tensor。
    - ``ShapeError`` - 如果输入和卷积核的维度大小不相同。
    - ``ShapeError`` - 如果输入的维度大小与 ``stride`` 之差不是2。

**代码示例**

..  code-block:: python

    import paddle
    import paddle.nn.functional as F

    x_var = paddle.randn((2, 3, 8, 8, 8), dtype='float32')
    w_var = paddle.randn((3, 6, 3, 3, 3), dtype='float32')

    y_var = F.conv3d_transpose(x_var, w_var)
    y_np = y_var.numpy()

    print(y_np.shape)
    # (2, 6, 10, 10, 10)

