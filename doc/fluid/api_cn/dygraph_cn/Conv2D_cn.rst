.. _cn_api_fluid_dygraph_Conv2D:

Conv2D
-------------------------------

.. py:class:: paddle.fluid.dygraph.Conv2D(name_scope, num_filters, filter_size, stride=1, padding=0, dilation=1, groups=None, param_attr=None, bias_attr=None, use_cudnn=True, act=None, dtype='float32')

二维卷积层（convolution2D layer）根据输入、滤波器参数（num_filters、filter_size）、步长（stride）、填充（padding）、膨胀系数（dilation）、组数（groups）参数计算输出。输入和输出是 ``NCHW`` 格式，N是批数据大小，C是特征图个数，H是特征图高度，W是特征图宽度。滤波器是 ``MCHW`` 格式，M是输出特征图个数，C是输入特征图个数，H是滤波器高度，W是滤波器宽度。如果组数大于1，C等于输入图像通道数除以组数的结果。详情请参考: `卷积 <http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/>`_ 。如果提供了偏移属性和激活函数类型，卷积的结果会与偏移相加，激活函数会作用在最终结果上。

对每个输入 ``X`` ，有等式：

.. math::

    Out = \sigma \left ( W * X + b \right )

其中：
    - :math:`X` ：输入值， ``NCHW`` 格式的Tensor
    - :math:`W` ：滤波器值， ``MCHW`` 格式的Tensor
    - :math:`*` ： 卷积操作
    - :math:`b` ：偏移值，2-D Tensor，维度为 ``[M,1]``
    - :math:`\sigma` ：激活函数
    - :math:`Out` ：输出值， ``Out`` 和 ``X`` 的维度可能不同

**输出维度计算示例**

- 输入：

  输入维度： :math:`( N,C_{in},H_{in},W_{in} )`

  滤波器维度： :math:`( C_{out},C_{in},H_{f},W_{f} )`

- 输出：

  输出维度： :math:`( N,C_{out},H_{out},W_{out} )`

- 其中

.. math::

    H_{out} = \frac{\left ( H_{in}+2*paddings[0]-\left ( dilations[0]*\left ( H_{f}-1 \right )+1 \right ) \right )}{strides[0]}+1

    W_{out} = \frac{\left ( W_{in}+2*paddings[1]-\left ( dilations[1]*\left ( W_{f}-1 \right )+1 \right ) \right )}{strides[1]}+1

参数：
    - **name_scope** (str) - 类的名称。
    - **num_fliters** (int) - 滤波器数量，和输出特征图个数相同。
    - **filter_size** (int|tuple) - 滤波器大小。如果 ``filter_size`` 是一个元组，则必须包含两个整型数，分别表示滤波器高度和宽度。否则，表示滤波器高度和宽度均为 ``filter_size`` 。
    - **stride** (int|tuple, 可选) - 步长大小。如果 ``stride`` 为元组，则必须包含两个整型数，分别表示垂直和水平滑动步长。否则，表示垂直和水平滑动步长均为 ``stride`` 。默认值：1。
    - **padding** (int|tuple, 可选) - 填充大小。如果 ``padding`` 为元组，则必须包含两个整型数，分别表示竖直和水平边界填充大小。否则，表示竖直和水平边界填充大小均为 ``padding`` 。默认值：0。
    - **dilation** (int|tuple, 可选) - 膨胀系数大小。如果 ``dialation`` 为元组，则必须包含两个整型数，分别表示垂直和水平膨胀系数。否则，表示垂直和水平膨胀系数均为 ``dialation`` 。默认值：1。
    - **groups** (int, 可选) - 二维卷积层的组数。根据Alex Krizhevsky的深度卷积神经网络（CNN）论文中的分组卷积：当group=2，滤波器的前一半仅和输入特征图的前一半连接。滤波器的后一半仅和输入特征图的后一半连接。默认值：1。
    - **param_attr** (ParamAttr, 可选) - 二维卷积层中可学习参数/权重的属性。如果设为None或者ParamAttr的一个属性，会根据 ``param_attr`` 创建ParamAttr对象。如果 ``param_attr`` 的初始化函数未设置，参数则初始化为 :math:`Normal(0.0,std)` ，并且std为 :math:`\frac{2.0}{filter\_elem\_num}^{0.5}` 。默认值：None。
    - **bias_attr** (ParamAttr|bool, 可选) - 二维卷积层的偏移参数属性。如果设为False，则没有偏移加到输出。如果设为None或者ParamAttr的一个属性，会根据 ``bias_attr`` 创建ParamAttr。如果 ``bias_attr`` 的初始化函数未设置，bias初始化为0。默认值：None。
    - **use_cudnn** （bool, 可选） - 是否用cudnn核，仅当下载cudnn库才有效。默认值：True。
    - **act** (str, 可选) - 激活函数类型，如果设为None，则未添加激活函数。默认值：None。


抛出异常：
  - ``ValueError`` - 如果输入shape和filter_size，stride,padding和groups不匹配。


**代码示例**

.. code-block:: python

    from paddle.fluid.dygraph.base import to_variable
    import paddle.fluid as fluid
    from paddle.fluid.dygraph import Conv2D
    import numpy as np

    data = np.random.uniform( -1, 1, [10, 3, 32, 32] ).astype('float32')
    with fluid.dygraph.guard():
        conv2d = Conv2D( "conv2d", 2, 3 )
        data = to_variable( data )
        conv = conv2d( data )

