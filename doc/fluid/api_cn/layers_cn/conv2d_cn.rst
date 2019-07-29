.. _cn_api_fluid_layers_conv2d:

conv2d
-------------------------------

.. py:function:: paddle.fluid.layers.conv2d(input, num_filters, filter_size, stride=1, padding=0, dilation=1, groups=None, param_attr=None, bias_attr=None, use_cudnn=True, act=None, name=None)

卷积二维层（convolution2D layer）根据输入、滤波器（filter）、步长（stride）、填充（padding）、dilations、一组参数计算输出。输入和输出是NCHW格式，N是批尺寸，C是通道数，H是特征高度，W是特征宽度。滤波器是MCHW格式，M是输出图像通道数，C是输入图像通道数，H是滤波器高度，W是滤波器宽度。如果组数大于1，C等于输入图像通道数除以组数的结果。详情请参考UFLDL's : `卷积 <http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/>`_ 。如果提供了bias属性和激活函数类型，bias会添加到卷积（convolution）的结果中相应的激活函数会作用在最终结果上。

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
    - **num_fliters** (int) - 滤波器数。和输出图像通道相同
    - **filter_size** (int|tuple|None) - 滤波器大小。如果filter_size是一个元组，则必须包含两个整型数，（filter_size，filter_size_W）。否则，滤波器为square
    - **stride** (int|tuple) - 步长(stride)大小。如果步长（stride）为元组，则必须包含两个整型数，（stride_H,stride_W）。否则，stride_H = stride_W = stride。默认：stride = 1
    - **padding** (int|tuple) - 填充（padding）大小。如果填充（padding）为元组，则必须包含两个整型数，（padding_H,padding_W)。否则，padding_H = padding_W = padding。默认：padding = 0
    - **dilation** (int|tuple) - 膨胀（dilation）大小。如果膨胀（dialation）为元组，则必须包含两个整型数，（dilation_H,dilation_W）。否则，dilation_H = dilation_W = dilation。默认：dilation = 1
    - **groups** (int) - 卷积二维层（Conv2D Layer）的组数。根据Alex Krizhevsky的深度卷积神经网络（CNN）论文中的成组卷积：当group=2，滤波器的前一半仅和输入通道的前一半连接。滤波器的后一半仅和输入通道的后一半连接。默认：groups = 1
    - **param_attr** (ParamAttr|None) - conv2d的可学习参数/权重的参数属性。如果设为None或者ParamAttr的一个属性，conv2d创建ParamAttr为param_attr。如果param_attr的初始化函数未设置，参数则初始化为 :math:`Normal(0.0,std)` ，并且std为 :math:`\frac{2.0}{filter\_elem\_num}^{0.5}` 。默认为None
    - **bias_attr** (ParamAttr|bool|None) - conv2d bias的参数属性。如果设为False，则没有bias加到输出。如果设为None或者ParamAttr的一个属性，conv2d创建ParamAttr为bias_attr。如果bias_attr的初始化函数未设置，bias初始化为0.默认为None
    - **use_cudnn** （bool） - 是否用cudnn核，仅当下载cudnn库才有效。默认：True
    - **act** (str) - 激活函数类型，如果设为None，则未添加激活函数。默认：None
    - **name** (str|None) - 该层名称（可选）。若设为None，则自动为该层命名。

返回：张量，存储卷积和非线性激活结果

返回类型：变量（Variable）

抛出异常:
  - ``ValueError`` - 如果输入shape和filter_size，stride,padding和group不匹配。

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(name='data', shape=[3, 32, 32], dtype='float32')
    conv2d = fluid.layers.conv2d(input=data, num_filters=2, filter_size=3, act="relu")











