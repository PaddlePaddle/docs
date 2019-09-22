.. _cn_api_fluid_layers_conv3d_transpose:

conv3d_transpose
-------------------------------

.. py:function:: paddle.fluid.layers.conv3d_transpose(input, num_filters, output_size=None, filter_size=None, padding=0, stride=1, dilation=1, groups=None, param_attr=None, bias_attr=None, use_cudnn=True, act=None, name=None)

三维转置卷积层（Convlution3D transpose layer)

该层根据输入（input）、滤波器（filter）和卷积核膨胀比例（dilations）、步长（stride）、填充（padding）来计算输出特征层大小或者通过output_size指定输出特征层大小。输入(Input)和输出(Output)为NCDHW格式。其中N为批尺寸，C为通道数（channel），D为特征深度，H为特征层高度，W为特征层宽度。转置卷积的计算过程相当于卷积的反向计算。转置卷积又被称为反卷积（但其实并不是真正的反卷积）。欲了解卷积转置层细节，请参考下面的说明和 参考文献_ 。如果参数bias_attr不为False, 转置卷积计算会添加偏置项。如果act不为None，则转置卷积计算之后添加相应的激活函数。

.. _参考文献: http://www.matthewzeiler.com/wp-content/uploads/2017/07/cvpr2010.pdf

输入 :math:`X` 和输出 :math:`Out` 函数关系如下：

.. math::
                        \\Out=\sigma (W*X+b)\\

其中：
    -  :math:`X` : 输入图像，具有NCDHW格式的张量（Tensor）
    -  :math:`W` : 滤波器，具有NCDHW格式的张量（Tensor）
    -  :math:`*` : 卷积操作（注意：转置卷积本质上的计算还是卷积）
    -  :math:`b` : 偏置（bias），二维张量，shape为 ``[M,1]``
    -  :math:`σ` : 激活函数
    -  :math:`Out` : 输出值， ``Out`` 和 ``X`` 的 shape可能不一样

**示例**

输入:

    输入的shape：:math:`（N,C_{in}, D_{in}, H_{in}, W_{in}）`

    滤波器的shape：:math:`（C_{in}, C_{out}, D_f, H_f, W_f）`



输出:

    输出的shape：:math:`（N,C_{out}, D_{out}, H_{out}, W_{out}）`


其中：

.. math::

    D'_{out}=(D_{in}-1)*strides[0]-2*paddings[0]+dilations[0]*(D_f-1)+1
    H'_{out}=(H_{in}-1)*strides[1]-2*paddings[1]+dilations[1]*(H_f-1)+1
    W'_{out}=(W_{in}-1)*strides[2]-2*paddings[2]+dilations[2]*(W_f-1)+1
    D_{out}\in[D'_{out},D'_{out} + strides[0])
    H_{out}\in[H'_{out},H'_{out} + strides[1])
    W_{out}\in[W'_{out},W'_{out} + strides[2])

注意：

如果output_size为None，则:math:`D_{out}` = :math:`D^\prime_{out}` , :math:`H_{out}` = :math:`H^\prime_{out}` , :math:`W_{out}` = :math:`W^\prime_{out}` ;否则，指定的output_size_depth（输出特征层的深度） :math:`D_{out}` 应当介于 :math:`D^\prime_{out}` 和 :math:`D^\prime_{out} + strides[0]` 之间（不包含 :math:`D^\prime_{out} + strides[0]` ），指定的output_size_height（输出特征层的高） :math:`H_{out}` 应当介于 :math:`H^\prime_{out}` 和 :math:`H^\prime_{out} + strides[1]` 之间（不包含 :math:`H^\prime_{out} + strides[1]` ）, 并且指定的output_size_width（输出特征层的宽） :math:`W_{out}` 应当介于 :math:`W^\prime_{out}` 和 :math:`W^\prime_{out} + strides[2]` 之间（不包含 :math:`W^\prime_{out} + strides[2]` ）。 

由于转置卷积可以当成是卷积的反向计算，而根据卷积的输入输出计算公式来说，不同大小的输入特征层可能对应着相同大小的输出特征层，所以对应到转置卷积来说，固定大小的输入特征层对应的输出特征层大小并不唯一。

如果指定了output_size， ``conv3d_transpose`` 可以自动计算滤波器的大小。

参数:
  - **input** （Variable）- 输入图像，格式为[N, C, D, H, W]的张量。
  - **num_filters** (int) - 滤波器（卷积核）的个数，与输出的图片的通道数相同。
  - **output_size** (int|tuple|None) - 输出图片的大小。如果output_size是一个元组，则必须包含三个整型数，（output_size_depth，output_size_height，output_size_width）。如果output_size=None，则内部会使用filter_size、padding和stride来计算output_size。如果output_size和filter_size是同时指定的，那么它们应满足上面的公式。
  - **filter_size** (int|tuple|None) - 滤波器大小。如果filter_size是一个元组，则必须包含三个整型数，（filter_size_depth，filter_size_height, filter_size_width）。否则，filter_size_depth = filter_size_height = filter_size_width = filter_size。如果filter_size=None，则必须指定output_size， ``conv2d_transpose`` 内部会根据output_size、padding和stride计算出滤波器大小。
  - **padding** (int|tuple) - 填充padding大小。输入的每个特征层四周填充的0的数量，padding_depth代表特征层深度两侧每一侧填充0的数量padding_height代表特征层上下两边每一边填充0的数量，padding_width代表特征层左右两边每一边填充0的数量。如果padding是一个元组，它必须包含三个整数(padding_depth，padding_height，padding_width)。否则，padding_depth = padding_height = padding_width = padding。默认：padding = 0。
  - **stride** (int|tuple) - 步长stride大小。滤波器和输入进行卷积计算时滑动的步长。如果stride是一个元组，那么元组的形式为(stride_depth，stride_height，stride_width)。否则，stride_depth = stride_height = stride_width = stride。默认：stride = 1。
  - **dilation** (int|tuple) - 膨胀比例dilation大小。空洞卷积时会指该参数，滤波器对输入进行卷积时，感受野里每相邻两个特征点之间的空洞信息，根据`可视化效果图<https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md>`_较好理解。如果膨胀比例dilation是一个元组，那么元组的形式为(dilation_depth，dilation_height， dilation_width)。否则，dilation_depth = dilation_height = dilation_width = dilation。默认:dilation= 1。
  - **groups** (int) - 三维转置卷积层的组数。从Alex Krizhevsky的CNN Deep论文中的群卷积中受到启发，当group=2时，输入和滤波器分别根据通道数量平均分为两组，第一组滤波器和第一组输入进行卷积计算，第二组滤波器和第二组输入进行卷积计算。默认：group = 1。
  - **param_attr** (ParamAttr|None) - conv3d_transpose 权重的参数属性。可以设置为None或者包含属性的ParamAttr类。如果设为包含属性的ParamAttr类，conv3d_transpose创建相应属性的ParamAttr类为param_attr参数。如果param_attr设置为None或者ParamAttr里的初始化函数未设置，那么默认使用Xavier初始化。默认：None。
  - **bias_attr** (ParamAttr|False|None) - conv3d_transpose 偏置参数属性。如果设置为False，则不会向输出单元添加偏置。如果设为包含属性的ParamAttr类，conv3d_transpose创建相应属性的ParamAttr类为bias_attr参数。如果bias_attr设置为None或者ParamAttr里的初始化函数未设置，bias将初始化为零。默认：None。
  - **use_cudnn** (bool) - 是否使用cudnn内核，只有已安装cudnn库时才有效。默认：True。
  - **act** (str) -  激活函数类型，如果设置为None，则不使用激活函数。默认：None。
  - **name** (str|None) - 该layer的名称(可选)。如果设置为None， 将自动命名该层。默认：True。

返回：如果未指定激活层，则返回转置卷积计算的结果，如果指定激活层，则返回转置卷积和激活计算之后的最终结果。

返回类型：Variable（Tensor），数据类型和输入相同。

**代码示例**：

..  code-block:: python

    import paddle.fluid as fluid
    import numpy as np
    data = fluid.layers.data(name='data', shape=[3, 12, 32, 32], dtype='float32')
    param_attr = fluid.ParamAttr(name='conv3d.weight', initializer=fluid.initializer.Xavier(uniform=False), learning_rate=0.001)
    res = fluid.layers.conv3d_transpose(input=data, output_size=(14, 66, 66), num_filters=2, filter_size=3, act="relu", param_attr=param_attr)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    x = np.random.rand(1, 3, 12, 32, 32).astype("float32")
    output = exe.run(feed={"data": x}, fetch_list=[res])
    print(output)
