.. _cn_api_fluid_dygraph_Conv3D:

Conv3D
-------------------------------

.. py:class:: paddle.fluid.dygraph.Conv3D(num_channels, num_filters, filter_size, stride=1, padding=0, dilation=1, groups=None, param_attr=None, bias_attr=None, use_cudnn=True, act=None, dtype="float32")





该接口用于构建 ``Conv3D`` 类的一个可调用对象，具体用法参照 ``代码示例`` 。3D卷积层（convolution3D layer）根据输入、滤波器（filter）、步长（stride）、填充（padding）、膨胀（dilations）、组数参数计算得到输出。输入和输出是[N, C, D, H, W]的多维tensor，其中N是批尺寸，C是通道数，D是特征深度，H是特征高度，W是特征宽度。卷积三维（Convlution3D）和卷积二维（Convlution2D）相似，但多了一维深度（depth）。如果提供了bias属性和激活函数类型，bias会添加到卷积（convolution）的结果中相应的激活函数会作用在最终结果上。

对每个输入X，有等式：

.. math::


    Out = \sigma \left ( W * X + b \right )

其中：
    - :math:`X`：输入值，NCDHW格式的张量（Tensor）
    - :math:`W`：滤波器值，MCDHW格式的张量（Tensor）
    - :math:`*`：卷积操作
    - :math:`b` ：Bias值，二维张量（Tensor），维度为 ``[M,1]``
    - :math:`\sigma`：激活函数
    - :math:`Out`：输出值，``Out`` 和 ``X`` 的维度可能不同

**示例**

- 输入：
    输入Tensor的维度：:math:`[N, C_{in}, D_{in}, H_{in}, W_{in}]`

    滤波器Tensor的维度：:math:`[C_{out}, C_{in}, D_f, H_f, W_f]`
- 输出：
    输出Tensor的维度：:math:`[N, C_{out}, D_{out}, H_{out}, W_{out}]`

其中

.. math::


    D_{out}&= \frac{(D_{in} + 2 * paddings[0] - (dilations[0] * (D_f - 1) + 1))}{strides[0]} + 1 \\
    H_{out}&= \frac{(H_{in} + 2 * paddings[1] - (dilations[1] * (H_f - 1) + 1))}{strides[1]} + 1 \\
    W_{out}&= \frac{(W_{in} + 2 * paddings[2] - (dilations[2] * (W_f - 1) + 1))}{strides[2]} + 1

参数
::::::::::::

    - **num_channels** (int) - 输入图像的通道数。
    - **num_fliters** (int) - 滤波器（卷积核）的个数。和输出图像通道相同。
    - **filter_size** (int|tuple) - 滤波器大小。如果filter_size是一个元组，则必须包含三个整型数，(filter_size_D, filter_size_H, filter_size_W)。如果filter_size是一个int型，则filter_size_depth = filter_size_height = filter_size_width = filter_size。
    - **stride** (int|tuple，可选) - 步长(stride)大小。滤波器和输入进行卷积计算时滑动的步长。如果步长（stride）为元组，则必须包含三个整型数，(stride_D, stride_H, stride_W)。否则，stride_D = stride_H = stride_W = stride。默认值为1。
    - **padding** (int|tuple，可选) - 填充（padding）大小。padding参数在输入特征层每边添加padding个0。如果填充（padding）为元组，则必须包含三个整型数，(padding_depth, padding_height, padding_width)。否则，padding_depth = padding_height = padding_width = padding。默认值为0。
    - **dilation** (int|tuple，可选) - 膨胀（dilation）大小。空洞卷积时会指该参数，滤波器对输入进行卷积时，感受野里每相邻两个特征点之间的空洞信息，根据 `可视化效果图  <https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md>`_ 较好理解。如果膨胀（dialation）为元组，则必须包含两个整型数，(dilation_D, dilation_H, dilation_W)。否则，dilation_D = dilation_H = dilation_W = dilation。默认值为1。
    - **groups** (int，可选) - 三维卷积层（Conv3D Layer）的组数。根据Alex Krizhevsky的深度卷积神经网络（CNN）论文中的成组卷积：当group=2，输入和滤波器分别根据通道数量平均分为两组，第一组滤波器和第一组输入进行卷积计算，第二组滤波器和第二组输入进行卷积计算。默认值为1。
    - **param_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **bias_attr** (ParamAttr，可选) - 指定偏置参数属性的对象。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr`。
    - **use_cudnn** （bool，可选） - 是否用cudnn内核，只有安装Paddle GPU版时才有效。默认值为True。
    - **act** (str，可选) - 激活函数类型，如果设为None，则未添加激活函数。默认值为None。
    - **dtype** (str，可选) - 数据类型，可以为"float32"或"float64"。默认值："float32"。


返回
::::::::::::
无

代码示例
::::::::::::

.. code-block:: python

    import paddle.fluid as fluid
    import numpy

    with fluid.dygraph.guard():
        data = numpy.random.random((5, 3, 12, 32, 32)).astype('float32')
        conv3d = fluid.dygraph.nn.Conv3D(
              'Conv3D', num_filters=2, filter_size=3, act="relu")
        ret = conv3d(fluid.dygraph.base.to_variable(data))


属性
::::::::::::
属性
::::::::::::
weight
'''''''''

本层的可学习参数，类型为 ``Parameter``

bias
'''''''''

本层的可学习偏置，类型为 ``Parameter``


