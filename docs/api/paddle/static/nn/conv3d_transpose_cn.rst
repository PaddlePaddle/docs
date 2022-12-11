.. _cn_api_fluid_layers_conv3d_transpose:

conv3d_transpose
-------------------------------


.. py:function:: paddle.static.nn.conv3d_transpose(input, num_filters, output_size=None, filter_size=None, padding=0, stride=1, dilation=1, groups=None, param_attr=None, bias_attr=None, use_cudnn=True, act=None, name=None, data_format='NCDHW')




三维转置卷积层（Convlution3D transpose layer)

该层根据输入（input）、滤波器（filter）和卷积核膨胀比例（dilations）、步长（stride）、填充（padding）来计算输出特征层大小或者通过 output_size 指定输出特征层大小。

输入(Input)和输出(Output)为 NCDHW 或者 NDHWC 格式。其中 N 为批尺寸，C 为通道数（channel），D 为特征深度，H 为特征层高度，W 为特征层宽度。

转置卷积的计算过程相当于卷积的反向计算。转置卷积又被称为反卷积（但其实并不是真正的反卷积）。欲了解卷积转置层细节，请参考下面的说明和论文细节。

如果参数 bias_attr 不为 False，转置卷积计算会添加偏置项。如果 act 不为 None，则转置卷积计算之后添加相应的激活函数。

论文参考：https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf

输入 :math:`X` 和输出 :math:`Out` 函数关系如下：

.. math::
                        \\Out=\sigma (W*X+b)\\

其中：

    -  :math:`X`：输入，具有 NCDHW 或 NDHWC 格式的 5-D Tensor；
    -  :math:`W`：滤波器，具有 NCDHW 格式的 5-D Tensor；
    -  :math:`*`：卷积操作（注意：转置卷积本质上的计算还是卷积）；
    -  :math:`b`：偏置（bias），2-D Tensor，形状为 ``[M,1]``；
    -  :math:`σ`：激活函数；
    -  :math:`Out`：输出值，NCDHW 或 NDHWC 格式的 5-D Tensor，和 ``X`` 的形状可能不同。

**示例**

输入：

    输入的 shape：:math:`（N,C_{in}, D_{in}, H_{in}, W_{in}）`

    滤波器的 shape：:math:`（C_{in}, C_{out}, D_f, H_f, W_f）`



输出：

    输出的 shape：:math:`（N,C_{out}, D_{out}, H_{out}, W_{out}）`


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

.. note::

如果 output_size 为 None，则 :math:`D_{out}` = :math:`D^\prime_{out}` , :math:`H_{out}` = :math:`H^\prime_{out}` , :math:`W_{out}` = :math:`W^\prime_{out}`；
否则，指定的 output_size_depth（输出特征层的深度） :math:`D_{out}` 应当介于 :math:`D^\prime_{out}` 和 :math:`D^\prime_{out} + strides[0]` 之间（不包含 :math:`D^\prime_{out} + strides[0]` ），指定的 output_size_height（输出特征层的高） :math:`H_{out}` 应当介于 :math:`H^\prime_{out}` 和 :math:`H^\prime_{out} + strides[1]` 之间（不包含 :math:`H^\prime_{out} + strides[1]` ），并且指定的 output_size_width（输出特征层的宽） :math:`W_{out}` 应当介于 :math:`W^\prime_{out}` 和 :math:`W^\prime_{out} + strides[2]` 之间（不包含 :math:`W^\prime_{out} + strides[2]` ）。

由于转置卷积可以当成是卷积的反向计算，而根据卷积的输入输出计算公式来说，不同大小的输入特征层可能对应着相同大小的输出特征层，所以对应到转置卷积来说，固定大小的输入特征层对应的输出特征层大小并不唯一。

如果指定了 output_size， ``conv3d_transpose`` 可以自动计算滤波器的大小。

参数
::::::::::::

  - **input** （Tensor）- 形状为 :math:`[N, C, D, H, W]` 或 :math:`[N, D, H, W, C]` 的 5-D Tensor，N 是批尺寸，C 是通道数，D 是特征深度，H 是特征高度，W 是特征宽度，数据类型：float32 或 float64。
  - **num_filters** (int) - 滤波器（卷积核）的个数，与输出的图片的通道数相同。
  - **output_size** (int|tuple，可选) - 输出图片的大小。如果 output_size 是一个元组，则必须包含三个整型数，（output_size_depth，output_size_height，output_size_width）。如果 output_size=None，则内部会使用 filter_size、padding 和 stride 来计算 output_size。如果 output_size 和 filter_size 是同时指定的，那么它们应满足上面的公式。默认：None。output_size 和 filter_size 不能同时为 None。
  - **filter_size** (int|tuple，可选) - 滤波器大小。如果 filter_size 是一个元组，则必须包含三个整型数，（filter_size_depth，filter_size_height, filter_size_width）。否则，filter_size_depth = filter_size_height = filter_size_width = filter_size。如果 filter_size=None，则必须指定 output_size， ``conv2d_transpose`` 内部会根据 output_size、padding 和 stride 计算出滤波器大小。默认：None。output_size 和 filter_size 不能同时为 None。
  - **padding** (int|list|tuple|str，可选) - 填充 padding 大小。padding 参数在输入特征层每边添加 ``dilation * (kernel_size - 1) - padding`` 个 0。如果它是一个字符串，可以是"VALID"或者"SAME"，表示填充算法，计算细节可参考上述 ``padding`` = "SAME"或  ``padding`` = "VALID" 时的计算公式。如果它是一个元组或列表，它可以有 3 种格式：

    - (1)包含 5 个二元组：当 ``data_format`` 为"NCDHW"时为 [[0,0], [0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]，当 ``data_format`` 为"NDHWC"时为[[0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]；
    - (2)包含 6 个整数值：[pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]；
    - (3)包含 3 个整数值：[pad_depth, pad_height, pad_width]，此时 pad_depth_front = pad_depth_back = pad_depth, pad_height_top = pad_height_bottom = pad_height, pad_width_left = pad_width_right = pad_width。若为一个整数，pad_depth = pad_height = pad_width = padding。默认值：0。

  - **stride** (int|tuple，可选) - 步长 stride 大小。滤波器和输入进行卷积计算时滑动的步长。如果 stride 是一个元组，那么元组的形式为(stride_depth，stride_height，stride_width)。否则，stride_depth = stride_height = stride_width = stride。默认：stride = 1。
  - **dilation** (int|tuple，可选) - 膨胀比例 dilation 大小。空洞卷积时会指该参数，滤波器对输入进行卷积时，感受野里每相邻两个特征点之间的空洞信息，根据 `可视化效果图 <https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md>`_ 较好理解。如果膨胀比例 dilation 是一个元组，那么元组的形式为(dilation_depth，dilation_height， dilation_width)。否则，dilation_depth = dilation_height = dilation_width = dilation。默认：dilation= 1。
  - **groups** (int，可选) - 三维转置卷积层的组数。从 Alex Krizhevsky 的 CNN Deep 论文中的群卷积中受到启发，当 group=2 时，输入和滤波器分别根据通道数量平均分为两组，第一组滤波器和第一组输入进行卷积计算，第二组滤波器和第二组输入进行卷积计算。默认：group = 1。
  - **param_attr** (ParamAttr，可选)：指定权重参数属性的对象。默认值为 None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。conv3d_transpose 算子默认的权重初始化是 Xavier。
  - **bias_attr** （ParamAttr|False，可选）- 指定偏置参数属性的对象。默认值为 None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。conv3d_transpose 算子默认的偏置初始化是 0.0。
  - **use_cudnn** (bool，可选) - 是否使用 cudnn 内核，只有已安装 cudnn 库时才有效。默认：True。
  - **act** (str，可选) -  激活函数类型，如果设置为 None，则不使用激活函数。默认：None。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
  - **data_format** (str，可选) - 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCDHW"和"NDHWC"。N 是批尺寸，C 是通道数，H 是特征高度，W 是特征宽度。默认值："NCDHW"。

返回
::::::::::::
5-D Tensor，数据类型与 ``input`` 一致。如果未指定激活层，则返回转置卷积计算的结果，如果指定激活层，则返回转置卷积和激活计算之后的最终结果。

代码示例
::::::::::::

COPY-FROM: paddle.static.nn.conv3d_transpose
