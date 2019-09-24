.. _cn_api_fluid_layers_conv3d:

conv3d
-------------------------------

.. py:function:: paddle.fluid.layers.conv3d(input, num_filters, filter_size, stride=1, padding=0, dilation=1, groups=None, param_attr=None, bias_attr=None, use_cudnn=True, act=None, name=None)

三维卷积层（convolution3D layer）根据输入、滤波器（filter）、步长（stride）、填充（padding）、膨胀比例（dilations），一组参数计算得到输出特征层大小。输入和输出是NCDHW格式，N是批尺寸，C是通道数，D是特征层深度，H是特征层高度，W是特征层宽度。3D卷积（Convlution3D）和2D卷积（Convlution2D）相似，但多了一维深度信息（depth）。如果bias_attr不为False，卷积（convolution）计算会添加偏置项。如果指定了激活函数类型，相应的激活函数会作用在最终结果上。

对每个输入X，有等式：

.. math::


    Out = \sigma \left ( W * X + b \right )

其中：
    - :math:`X` ：输入值，NCDHW格式的张量（Tensor）
    - :math:`W` ：滤波器值，MCDHW格式的张量（Tensor）
    - :math:`*` ： 卷积操作
    - :math:`b` ：Bias值，二维张量（Tensor），形为 ``[M,1]``
    - :math:`\sigma` ：激活函数
    - :math:`Out` ：输出值, 和 ``X`` 的形状可能不同

**示例**

- 输入：

  输入shape： :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`

  滤波器shape： :math:`(C_{out}, C_{in}, D_f, H_f, W_f)`

- 输出：

  输出shape： :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`

其中

.. math::


    D_{out}&= \frac{(D_{in} + 2 * paddings[0] - (dilations[0] * (D_f - 1) + 1))}{strides[0]} + 1 \\
    H_{out}&= \frac{(H_{in} + 2 * paddings[1] - (dilations[1] * (H_f - 1) + 1))}{strides[1]} + 1 \\
    W_{out}&= \frac{(W_{in} + 2 * paddings[2] - (dilations[2] * (W_f - 1) + 1))}{strides[2]} + 1

参数：
    - **input** (Variable) - 格式为[N,C,D,H,W]格式的输入图像。
    - **num_fliters** (int) - 滤波器（卷积核）的个数。和输出图像通道相同。
    - **filter_size** (int|tuple) - 滤波器大小。如果filter_size是一个元组，则必须包含三个整型数，(filter_size_depth, filter_size_height, filter_size_width)。如果filter_size是一个int型，则filter_size_depth = filter_size_height = filter_size_width = filter_size。
    - **stride** (int|tuple) - 步长(stride)大小。滤波器和输入进行卷积计算时滑动的步长。如果步长（stride）为元组，则必须包含三个整型数， (stride_depth, stride_height, stride_width)。否则，stride_depth = stride_height = stride_width = stride。默认：stride = 1。
    - **padding** (int|tuple) - 填充（padding）大小。padding参数在输入特征层每边添加 :math::``dilation * (kernel_size - 1) - padding`` 个0。如果填充（padding）为元组，则必须包含三个整型数，(padding_depth, padding_height, padding_width)。否则， padding_depth = padding_height = padding_width = padding。默认：padding = 0。
    - **dilation** (int|tuple) - 膨胀比例（dilation）大小。空洞卷积时会指该参数，滤波器对输入进行卷积时，感受野里每相邻两个特征点之间的空洞信息，根据`可视化效果图<https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md>`_较好理解。如果膨胀比例（dialation）为元组，则必须包含三个整型数， (dilation_depth, dilation_height, dilation_width)。否则，dilation_depth = dilation_height = dilation_width = dilation。默认：dilation = 1。
    - **groups** (int) - 三维卷积层（conv3d layer）的组数。根据Alex Krizhevsky的深度卷积神经网络（CNN）论文中的成组卷积：当group=2，输入和滤波器分别根据通道数量平均分为两组，第一组滤波器和第一组输入进行卷积计算，第二组滤波器和第二组输入进行卷积计算。默认：groups = 1。
    - **param_attr** (ParamAttr|None) - conv3d 权重的参数属性。可以设置为None或者包含属性的ParamAttr类。如果设置为包含属性的ParamAttr类，conv3d创建相应属性的ParamAttr类为param_attr参数。如果param_attr设置为None或者ParamAttr里的初始化函数未设置，参数初始化为 :math:`Normal(0.0,std)`，并且std为 :math:`\left ( \frac{2.0}{filter\_elem\_num} \right )^{0.5}` 。默认：None。
    - **bias_attr** (ParamAttr|False|None) - conv3d 偏置参数属性。如果设为False，则卷积不会加上偏置项。如果设为包含属性的ParamAttr类，conv3d创建相应属性的ParamAttr类为bias_attr参数。如果bias_attr设为None或者ParamAttr的初始化函数未设置，bias参数初始化为0。默认：None。
    - **use_cudnn** （bool） - 是否用cudnn核，仅当下载cudnn库才有效。默认：True。
    - **act** (str) - 激活函数类型，如果设为None，则未添加激活函数。默认：None。
    - **name** (str|None) - 该层名称（可选）。若设为None，则自动为该层命名。默认：None。

返回：如果未指定激活层，则返回卷积计算的结果，如果指定激活层，则返回卷积和激活计算之后的最终结果。

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np
    data = fluid.layers.data(name='data', shape=[3, 12, 32, 32], dtype='float32')
    param_attr = fluid.ParamAttr(name='conv3d.weight', initializer=fluid.initializer.Xavier(uniform=False), learning_rate=0.001)
    res = fluid.layers.conv3d(input=data, num_filters=2, filter_size=3, act="relu", param_attr=param_attr)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    x = np.random.rand(1, 3, 12, 32, 32).astype("float32")
    output = exe.run(feed={"data": x}, fetch_list=[res])
    print(output)


