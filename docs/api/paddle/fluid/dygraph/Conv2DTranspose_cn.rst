.. _cn_api_fluid_dygraph_Conv2DTranspose:

Conv2DTranspose
-------------------------------

.. py:class:: paddle.fluid.dygraph.Conv2DTranspose(num_channels, num_filters, filter_size, output_size=None, padding=0, stride=1, dilation=1, groups=None, param_attr=None, bias_attr=None, use_cudnn=True, act=None, dtype="float32")




该接口用于构建 ``Conv2DTranspose`` 类的一个可调用对象，具体用法参照 ``代码示例``。其将在神经网络中构建一个二维卷积转置层（Convlution2D Transpose Layer），其根据输入（input）、滤波器参数（num_filters、filter_size）、步长（stride）、填充（padding）、膨胀系数（dilation）、组数（groups）来计算得到输出特征图。输入和输出是 ``NCHW`` 格式，N是批数据大小，C是特征图个数，H是特征图高度，W是特征图宽度。滤波器的维度是 [M, C, H, W] ，M是输入特征图个数，C是输出特征图个数，H是滤波器高度，W是滤波器宽度。如果组数大于1，C等于输入特征图个数除以组数的结果。如果提供了偏移属性和激活函数类型，卷积的结果会和偏移相加，激活函数会作用在最终结果上。转置卷积的计算过程相当于卷积的反向计算，转置卷积又被称为反卷积（但其实并不是真正的反卷积）。详情请参考：`Conv2DTranspose <https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf>`_ 。

输入 ``X`` 和输出 ``Out`` 的函数关系如下：

.. math::
                        Out=\sigma (W*X+b)\\

其中：
    - :math:`X`：输入特征图，``NCHW`` 格式的 ``Tensor``
    - :math:`W`：滤波器，维度为 [M, C, H, W] 的 ``Tensor``
    - :math:`*`：卷积操作
    - :math:`b`：偏移值，2-D ``Tensor``，维度为 ``[M,1]``
    - :math:`\sigma`：激活函数
    - :math:`Out`：输出值，``Out`` 和 ``X`` 的维度可能不同

**输出维度计算示例**

- 输入：

  输入维度：:math:`(N,C_{in},H_{in},W_{in})`

  滤波器维度：:math:`(C_{in},C_{out},H_{f},W_{f})`

- 输出：

  输出维度：:math:`(N,C_{out},H_{out},W_{out})`

- 其中

.. math::

        & H'_{out} = (H_{in}-1)*strides[0]-2*paddings[0]+dilations[0]*(H_f-1)+1
        
        & W'_{out} = (W_{in}-1)*strides[1]-2*paddings[1]+dilations[1]*(W_f-1)+1
        
        & H_{out}\in[H'_{out},H'_{out} + strides[0])
        
        & W_{out}\in[W'_{out},W'_{out} + strides[1])

参数
::::::::::::

    - **num_channels** (int) - 输入图像的通道数。
    - **num_filters** (int) - 滤波器的个数，和输出特征图个数相同。
    - **filter_size** (int|tuple) - 滤波器大小。如果 ``filter_size`` 是一个元组，则必须包含两个整型数，分别表示滤波器高度和宽度。否则，表示滤波器高度和宽度均为 ``filter_size`` 。
    - **output_size** (int|tuple，可选) - 输出特征图的大小。如果 ``output_size`` 是一个元组，则必须包含两个整型数，分别表示特征图高度和宽度。如果 ``output_size`` 是整型，表示特征图高度和宽度均为 ``output_size``。如果 ``output_size`` 为None，则会根据 ``filter_size`` 、 ``padding`` 和 ``stride`` 来计算 ``output_size``。如果 ``output_size`` 和 ``filter_size`` 同时指定，那么它们应满足上面的公式。默认值：None。
    - **padding** (int|tuple，可选) - 填充大小。如果 ``padding`` 为元组，则必须包含两个整型数，分别表示竖直和水平边界填充大小。否则，表示竖直和水平边界填充大小均为 ``padding``。默认值：0。
    - **stride** (int|tuple，可选) - 步长大小。如果 ``stride`` 为元组，则必须包含两个整型数，分别表示垂直和水平滑动步长。否则，表示垂直和水平滑动步长均为 ``stride``。默认值：1。
    - **dilation** (int|tuple，可选) - 膨胀系数大小。如果 ``dialation`` 为元组，则必须包含两个整型数，分别表示垂直和水平膨胀系数。否则，表示垂直和水平膨胀系数均为 ``dialation``。默认值：1。
    - **groups** (int，可选) - 二维卷积层的组数。根据Alex Krizhevsky的深度卷积神经网络（CNN）论文中的分组卷积：当group=2，滤波器的前一半仅和输入特征图的前一半连接。滤波器的后一半仅和输入特征图的后一半连接。默认值：1。
    - **param_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **bias_attr** (ParamAttr|bool，可选) - 指定偏置参数属性的对象。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **use_cudnn** (bool，可选) - 是否使用cudnn内核，只有已安装cudnn库时才有效。默认值：True。
    - **act** (str，可选) -  应用于输出上的激活函数，如tanh、softmax、sigmoid，relu等，支持列表请参考 :ref:`api_guide_activations`，默认值：None。
    - **dtype** (str，可选) - 数据类型，可以为"float32"或"float64"。默认值："float32"。

返回
::::::::::::
无

代码示例
::::::::::::

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np

    with fluid.dygraph.guard():
        data = np.random.random((3, 32, 32, 5)).astype('float32')
        conv2DTranspose = fluid.dygraph.nn.Conv2DTranspose(
              num_channels=32, num_filters=2, filter_size=3)
        ret = conv2DTranspose(fluid.dygraph.base.to_variable(data))

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

