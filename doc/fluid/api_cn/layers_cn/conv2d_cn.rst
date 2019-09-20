.. _cn_api_fluid_layers_conv2d:

conv2d
-------------------------------

.. py:function:: paddle.fluid.layers.conv2d(input, num_filters, filter_size, stride=1, padding=0, dilation=1, groups=None, param_attr=None, bias_attr=None, use_cudnn=True, act=None, name=None)

二维卷积层（convolution2D layer）根据输入、滤波器（filter）、步长（stride）、填充（padding）、dilations、一组参数计算输出。输入和输出是NCHW格式，N是批尺寸，C是通道数，H是特征层的高度，W是特征层的宽度。滤波器是MCHW格式，M是输出图像通道数，C是输入图像通道数，H是滤波器高度，W是滤波器宽度。如果组数大于1，C等于输入图像通道数除以组数的结果。详情请参考UFLDL's : `卷积 <http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/>`_ 。如果提供了bias属性，bias会添加到偏差值到卷积（convolution）的结果中。如果指定了激活函数类型，相应的激活函数会作用在最终结果上。

对每个输入X，有等式：

.. math::

    Out = \sigma \left ( W * X + b \right )

其中：
    - :math:`X` ：输入值，NCHW格式的张量（Tensor）
    - :math:`W` ：滤波器值，MCHW格式的张量（Tensor）
    - :math:`*` ： 卷积操作
    - :math:`b` ：Bias值，二维张量（Tensor），shape为 ``[M,1]``
    - :math:`\sigma` ：激活函数
    - :math:`Out` ：输出值，``Out`` 和 ``X`` 的shape可能不同

**示例**

- 输入：

  输入shape：:math:`( N,C_{in},H_{in},W_{in} )`

  滤波器shape： :math:`( C_{out},C_{in},H_{f},W_{f} )`

- 输出：

  输出shape： :math:`( N,C_{out},H_{out},W_{out} )`

其中

.. math::

    H_{out} = \frac{\left ( H_{in}+2*paddings[0]-\left ( dilations[0]*\left ( H_{f}-1 \right )+1 \right ) \right )}{strides[0]}+1

    W_{out} = \frac{\left ( W_{in}+2*paddings[1]-\left ( dilations[1]*\left ( W_{f}-1 \right )+1 \right ) \right )}{strides[1]}+1

参数：
    - **input** (Variable) - 格式为[N,C,H,W]格式的输入图像
    - **num_filters** (int) - 滤波器数。和输出图像通道相同
    - **filter_size** (int|tuple) - 滤波器大小。如果filter_size是一个元组，则必须包含两个整型数，（filter_size_height，filter_size_width）。如果filter_size是一个int型，则filter_size_height = filter_size_width = filter_size。
    - **stride** (int|tuple) - 步长stride大小。如果步长stride是一个元组，则必须包含两个整型数，（stride_height,stride_width）。否则，stride_height = stride_width = stride。默认：stride = 1
    - **padding** (int|tuple) - 填充padding大小。如果填充padding为元组，则必须包含两个整型数，（padding_height,padding_width)。否则，padding_height = padding_width = padding。默认：padding = 0
    - **dilation** (int|tuple) - 膨胀比例dilation大小。如果膨胀比例dialation为元组，则必须包含两个整型数，（dilation_height,dilation_width）。否则，dilation_height = dilation_width = dilation。默认：dilation = 1
    - **groups** (int) - 二维卷积层（Conv2D Layer）的组数。根据Alex Krizhevsky的深度卷积神经网络（CNN）论文中的成组卷积：当group=2，输入和滤波器分别根据通道数量平均分为两组，第一组滤波器和第一组输入进行卷积计算。第二组滤波器和第二组输入进行卷积计算。默认：groups = 1
    - **param_attr** (ParamAttr|None) - conv2d 权重的参数属性，可以设置为None或者包含属性的ParamAttr类。如果设为包含属性的ParamAttr类，conv2d创建相应属性的ParamAttr类为param_attr参数。如果设置为None或者初始化函数未设置，参数默认初始化为 :math:`Normal(0.0,std)` ，并且std为 :math:`\frac{2.0}{filter\_elem\_num}^{0.5}` 。默认为None
    - **bias_attr** (ParamAttr|False|None) - conv2d 偏置参数属性。如果设为False，则卷积不会加上偏置项。如果设为包含属性的ParamAttr类，conv2d创建相应属性的ParamAttr类为bias_attr参数。如果设置为None或者bias_attr的初始化函数未设置，bias默认初始化为0.默认为None
    - **use_cudnn** （bool） - 是否用cudnn核，仅当下载cudnn库才有效。默认：True
    - **act** (str) - 激活函数类型，如果设为None，则未添加激活函数。默认：None
    - **name** (str|None) - 该层名称（可选）。若设为None，则自动为该层命名。

返回：张量，最终输出结果

返回类型：变量（Variable）

抛出异常:
  - ``ValueError`` - 如果输入shape和filter_size，stride,padding和group不匹配。

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np
    data = fluid.layers.data(name='data', shape=[3, 32, 32], dtype='float32')
    res = fluid.layers.conv2d(input=data, num_filters=2, filter_size=3, act="relu")
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_progrm())
    x = np.random.rand(1, 3, 32, 32).astype("float32")
    output = exe.run(feed={"data", x}, fetch_list=[res])
    print(output)


