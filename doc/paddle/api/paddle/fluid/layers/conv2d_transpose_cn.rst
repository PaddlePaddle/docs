.. _cn_api_fluid_layers_conv2d_transpose:

conv2d_transpose
-------------------------------


.. py:function:: paddle.fluid.layers.conv2d_transpose(input, num_filters, output_size=None, filter_size=None, padding=0, stride=1, dilation=1, groups=None, param_attr=None, bias_attr=None, use_cudnn=True, act=None, name=None, data_format='NCHW')




二维转置卷积层（Convlution2D transpose layer）

该层根据输入（input）、滤波器（filter）和卷积核膨胀比例（dilations）、步长（stride）、填充（padding）来计算输出特征层大小或者通过output_size指定输出特征层大小。输入(Input)和输出(Output)为NCHW或NHWC格式，其中N为批尺寸，C为通道数（channel），H为特征层高度，W为特征层宽度。滤波器是MCHW格式，M是输出图像通道数，C是输入图像通道数，H是滤波器高度，W是滤波器宽度。如果组数大于1，C等于输入图像通道数除以组数的结果。转置卷积的计算过程相当于卷积的反向计算。转置卷积又被称为反卷积（但其实并不是真正的反卷积）。欲了解转置卷积层细节，请参考下面的说明和 参考文献_ 。如果参数bias_attr不为False, 转置卷积计算会添加偏置项。如果act不为None，则转置卷积计算之后添加相应的激活函数。

.. _参考文献: https://arxiv.org/pdf/1603.07285.pdf


输入 :math:`X` 和输出 :math:`Out` 函数关系如下：

.. math::
                        Out=\sigma (W*X+b)\\

其中：
    -  :math:`X` : 输入，具有NCHW或NHWC格式的4-D Tensor
    -  :math:`W` : 滤波器，具有NCHW格式的4-D Tensor
    -  :math:`*` : 卷积计算（注意：转置卷积本质上的计算还是卷积）
    -  :math:`b` : 偏置（bias），2-D Tensor，形状为 ``[M,1]``
    -  :math:`σ` : 激活函数
    -  :math:`Out` : 输出值，NCHW或NHWC格式的4-D Tensor， 和 ``X`` 的形状可能不同

**示例**

- 输入：

    输入Tensor的形状： :math:`（N，C_{in}， H_{in}， W_{in}）`

    滤波器的形状 ： :math:`（C_{in}, C_{out}, H_f, W_f）`

- 输出：

    输出Tensor的形状 ： :math:`（N，C_{out}, H_{out}, W_{out}）`

其中

.. math::

        & H'_{out} = (H_{in}-1)*strides[0] - pad\_height\_top - pad\_height\_bottom + dilations[0]*(H_f-1)+1\\
        & W'_{out} = (W_{in}-1)*strides[1]- pad\_width\_left - pad\_width\_right + dilations[1]*(W_f-1)+1 \\
        & H_{out}\in[H'_{out},H'_{out} + strides[0])\\
        & W_{out}\in[W'_{out},W'_{out} + strides[1])\\

如果 ``padding`` = "SAME":

.. math::
   & H'_{out} = \frac{(H_{in} + stride[0] - 1)}{stride[0]}\\
   & W'_{out} = \frac{(W_{in} + stride[1] - 1)}{stride[1]}\\

如果 ``padding`` = "VALID":

.. math::
    & H'_{out} = (H_{in}-1)*strides[0] + dilations[0]*(H_f-1)+1\\
    & W'_{out} = (W_{in}-1)*strides[1] + dilations[1]*(W_f-1)+1 \\

注意：

如果output_size为None，则 :math:`H_{out}` = :math:`H^\prime_{out}` , :math:`W_{out}` = :math:`W^\prime_{out}` ;否则，指定的output_size_height（输出特征层的高） :math:`H_{out}` 应当介于 :math:`H^\prime_{out}` 和 :math:`H^\prime_{out} + strides[0]` 之间（不包含 :math:`H^\prime_{out} + strides[0]` ）, 并且指定的output_size_width（输出特征层的宽） :math:`W_{out}` 应当介于 :math:`W^\prime_{out}` 和 :math:`W^\prime_{out} + strides[1]` 之间（不包含 :math:`W^\prime_{out} + strides[1]` ）。

由于转置卷积可以当成是卷积的反向计算，而根据卷积的输入输出计算公式来说，不同大小的输入特征层可能对应着相同大小的输出特征层，所以对应到转置卷积来说，固定大小的输入特征层对应的输出特征层大小并不唯一。

如果指定了output_size， ``conv2d_transpose`` 可以自动计算滤波器的大小。

参数:
  - **input** （Variable）- 形状为 :math:`[N, C, H, W]` 或 :math:`[N, H, W, C]` 的4-D Tensor，N是批尺寸，C是通道数，H是特征高度，W是特征宽度。数据类型：float32或float64。
  - **num_filters** (int) - 滤波器（卷积核）的个数，与输出图片的通道数相同。
  - **output_size** (int|tuple，可选) - 输出图片的大小。如果output_size是一个元组，则必须包含两个整型数，（output_size_height，output_size_width）。如果output_size=None，则内部会使用filter_size、padding和stride来计算output_size。如果output_size和filter_size是同时指定的，那么它们应满足上面的公式。默认：None。output_size和filter_size不能同时为None。
  - **filter_size** (int|tuple，可选) - 滤波器大小。如果filter_size是一个元组，则必须包含两个整型数，（filter_size_height, filter_size_width）。否则，filter_size_height = filter_size_width = filter_size。如果filter_size=None，则必须指定output_size， ``conv2d_transpose`` 内部会根据output_size、padding和stride计算出滤波器大小。默认：None。output_size和filter_size不能同时为None。
  - **padding** (int|list|tuple|str，可选) - 填充padding大小。padding参数在输入特征层每边添加 ``dilation * (kernel_size - 1) - padding`` 个0。如果它是一个字符串，可以是"VALID"或者"SAME"，表示填充算法，计算细节可参考上述 ``padding`` = "SAME"或  ``padding`` = "VALID" 时的计算公式。如果它是一个元组或列表，它可以有3种格式：(1)包含4个二元组：当 ``data_format`` 为"NCHW"时为 [[0,0], [0,0], [padding_height_top, padding_height_bottom], [padding_width_left, padding_width_right]]，当 ``data_format`` 为"NHWC"时为[[0,0], [padding_height_top, padding_height_bottom], [padding_width_left, padding_width_right], [0,0]]；(2)包含4个整数值：[padding_height_top, padding_height_bottom, padding_width_left, padding_width_right]；(3)包含2个整数值：[padding_height, padding_width]，此时padding_height_top = padding_height_bottom = padding_height， padding_width_left = padding_width_right = padding_width。若为一个整数，padding_height = padding_width = padding。默认值：0。
  - **stride** (int|tuple，可选) - 步长stride大小。滤波器和输入进行卷积计算时滑动的步长。如果stride是一个元组，则必须包含两个整型数，形式为(stride_height，stride_width)。否则，stride_height = stride_width = stride。默认：stride = 1。
  - **dilation** (int|tuple，可选) - 膨胀比例(dilation)大小。空洞卷积时会指该参数，滤波器对输入进行卷积时，感受野里每相邻两个特征点之间的空洞信息，根据 `可视化效果图 <https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md>`_ 较好理解。如果膨胀比例dilation是一个元组，那么元组必须包含两个整型数，形式为(dilation_height, dilation_width)。否则，dilation_height = dilation_width = dilation。默认：dilation= 1。
  - **groups** (int，可选) - 二维转置卷积层的组数。从Alex Krizhevsky的CNN Deep论文中的群卷积中受到启发，当group=2时，输入和滤波器分别根据通道数量平均分为两组，第一组滤波器和第一组输入进行卷积计算，第二组滤波器和第二组输入进行卷积计算。默认：group = 1。
  - **param_attr** (ParamAttr，可选) ：指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。conv2d_transpose算子默认的权重初始化是Xavier。
  - **bias_attr** （ParamAttr|False，可选）- 指定偏置参数属性的对象。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。conv2d_transpose算子默认的偏置初始化是0.0。
  - **use_cudnn** (bool，可选) - 是否使用cudnn内核，只有已安装cudnn库时才有效。默认：True。
  - **act** (str，可选) -  激活函数类型，如果设置为None，则不使用激活函数。默认：None。
  - **name** (str，可选) – 具体用法请参见 :ref:`cn_api_guide_Name` ，一般无需设置，默认值为None。
  - **data_format** (str，可选) - 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCHW"和"NHWC"。N是批尺寸，C是通道数，H是特征高度，W是特征宽度。默认值："NCHW"。

返回：4-D Tensor，数据类型与 ``input`` 一致。如果未指定激活层，则返回转置卷积计算的结果，如果指定激活层，则返回转置卷积和激活计算之后的最终结果。

返回类型：Variable

抛出异常:
    -  ``ValueError`` : 如果输入的shape、filter_size、stride、padding和groups不匹配，抛出ValueError
    -  ``ValueError`` - 如果 ``data_format`` 既不是"NCHW"也不是"NHWC"。
    -  ``ValueError`` - 如果 ``padding`` 是字符串，既不是"SAME"也不是"VALID"。
    -  ``ValueError`` - 如果 ``padding`` 含有4个二元组，与批尺寸对应维度的值不为0或者与通道对应维度的值不为0。
    -  ``ValueError`` - 如果 ``output_size`` 和 ``filter_size`` 同时为None。
    -  ``ShapeError`` - 如果输入不是4-D Tensor。
    -  ``ShapeError`` - 如果输入和滤波器的维度大小不相同。
    -  ``ShapeError`` - 如果输入的维度大小与 ``stride`` 之差不是2。

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    import numpy as np
    data = fluid.layers.data(name='data', shape=[3, 32, 32], dtype='float32')
    param_attr = fluid.ParamAttr(name='conv2d.weight', initializer=fluid.initializer.Xavier(uniform=False), learning_rate=0.001)
    res = fluid.layers.conv2d_transpose(input=data, num_filters=2, filter_size=3, act="relu", param_attr=param_attr)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    x = np.random.rand(1, 3, 32, 32).astype("float32")
    output = exe.run(feed={"data": x}, fetch_list=[res])
    print(output)


