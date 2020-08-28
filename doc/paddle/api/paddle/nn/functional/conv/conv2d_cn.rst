.. _cn_api_fluid_layers_conv2d:

conv2d
-------------------------------


.. py:function:: paddle.fluid.layers.conv2d(input, num_filters, filter_size, stride=1, padding=0, dilation=1, groups=None, param_attr=None, bias_attr=None, use_cudnn=True, act=None, name=None, data_format="NCHW")

:api_attr: 声明式编程模式（静态图)



该OP是二维卷积层（convolution2D layer），根据输入、滤波器、步长（stride）、填充（padding）、膨胀比例（dilations）一组参数计算输出特征层大小。输入和输出是NCHW或NHWC格式，其中N是批尺寸，C是通道数，H是特征高度，W是特征宽度。滤波器是MCHW格式，M是输出图像通道数，C是输入图像通道数，H是滤波器高度，W是滤波器宽度。如果组数(groups)大于1，C等于输入图像通道数除以组数的结果。详情请参考UFLDL's : `卷积 <http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/>`_ 。如果bias_attr不为False，卷积计算会添加偏置项。如果指定了激活函数类型，相应的激活函数会作用在最终结果上。

对每个输入X，有等式：

.. math::

    Out = \sigma \left ( W * X + b \right )

其中：
    - :math:`X` ：输入值，NCHW或NHWC格式的4-D Tensor
    - :math:`W` ：滤波器值，MCHW格式的4-D Tensor
    - :math:`*` ：卷积操作
    - :math:`b` ：偏置值，2-D Tensor，形状为 ``[M,1]``
    - :math:`\sigma` ：激活函数
    - :math:`Out` ：输出值，NCHW或NHWC格式的4-D Tensor， 和 ``X`` 的形状可能不同

**示例**

- 输入：

  输入形状：:math:`（N,C_{in},H_{in},W_{in}）`

  滤波器形状： :math:`（C_{out},C_{in},H_{f},W_{f}）`

- 输出：

  输出形状： :math:`（N,C_{out},H_{out},W_{out}）`

其中

.. math::

    H_{out} &= \frac{\left ( H_{in} + padding\_height\_top + padding\_height\_bottom-\left ( dilation[0]*\left ( H_{f}-1 \right )+1 \right ) \right )}{stride[0]}+1

    W_{out} &= \frac{\left ( W_{in} + padding\_width\_left + padding\_width\_right -\left ( dilation[1]*\left ( W_{f}-1 \right )+1 \right ) \right )}{stride[1]}+1

如果 ``padding`` = "SAME":

.. math::
    H_{out} = \frac{(H_{in} + stride[0] - 1)}{stride[0]}

.. math::
    W_{out} = \frac{(W_{in} + stride[1] - 1)}{stride[1]}

如果 ``padding`` = "VALID":

.. math::
    H_{out} = \frac{\left ( H_{in} -\left ( dilation[0]*\left ( H_{f}-1 \right )+1 \right ) \right )}{stride[0]}+1

    W_{out} = \frac{\left ( W_{in} -\left ( dilation[1]*\left ( W_{f}-1 \right )+1 \right ) \right )}{stride[1]}+1

参数：
    - **input** (Variable) - 形状为 :math:`[N, C, H, W]` 或 :math:`[N, H, W, C]` 的4-D Tensor，N是批尺寸，C是通道数，H是特征高度，W是特征宽度，数据类型为float16, float32或float64。
    - **num_filters** (int) - 滤波器（卷积核）的个数。和输出图像通道相同。
    - **filter_size** (int|list|tuple) - 滤波器大小。如果它是一个列表或元组，则必须包含两个整数值：（filter_size_height，filter_size_width）。若为一个整数，filter_size_height = filter_size_width = filter_size。
    - **stride** (int|list|tuple，可选) - 步长大小。滤波器和输入进行卷积计算时滑动的步长。如果它是一个列表或元组，则必须包含两个整型数：（stride_height,stride_width）。若为一个整数，stride_height = stride_width = stride。默认值：1。
    - **padding** (int|list|tuple|str，可选) - 填充大小。如果它是一个字符串，可以是"VALID"或者"SAME"，表示填充算法，计算细节可参考上述 ``padding`` = "SAME"或  ``padding`` = "VALID" 时的计算公式。如果它是一个元组或列表，它可以有3种格式：(1)包含4个二元组：当 ``data_format`` 为"NCHW"时为 [[0,0], [0,0], [padding_height_top, padding_height_bottom], [padding_width_left, padding_width_right]]，当 ``data_format`` 为"NHWC"时为[[0,0], [padding_height_top, padding_height_bottom], [padding_width_left, padding_width_right], [0,0]]；(2)包含4个整数值：[padding_height_top, padding_height_bottom, padding_width_left, padding_width_right]；(3)包含2个整数值：[padding_height, padding_width]，此时padding_height_top = padding_height_bottom = padding_height， padding_width_left = padding_width_right = padding_width。若为一个整数，padding_height = padding_width = padding。默认值：0。
    - **dilation** (int|list|tuple，可选) - 膨胀比例大小。空洞卷积时会使用该参数，滤波器对输入进行卷积时，感受野里每相邻两个特征点之间的空洞信息。如果膨胀比例为列表或元组，则必须包含两个整型数：（dilation_height,dilation_width）。若为一个整数，dilation_height = dilation_width = dilation。默认值：1。
    - **groups** (int，可选) - 二维卷积层的组数。根据Alex Krizhevsky的深度卷积神经网络（CNN）论文中的成组卷积：当group=n，输入和滤波器分别根据通道数量平均分为n组，第一组滤波器和第一组输入进行卷积计算，第二组滤波器和第二组输入进行卷积计算，……，第n组滤波器和第n组输入进行卷积计算。默认值：1。
    - **param_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **bias_attr** （ParamAttr|bool，可选）- 指定偏置参数属性的对象。若 ``bias_attr`` 为bool类型，只支持为False，表示没有偏置参数。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **use_cudnn** （bool，可选）- 是否使用cudnn内核。只有已安装cudnn库时才有效。默认值：True。
    - **act** (str，可选) - 激活函数类型， 如tanh、softmax、sigmoid，relu等，支持列表请参考 :ref:`api_guide_activations` 。如果设为None，则未添加激活函数。默认值：None。
    - **name** (str，可选) – 具体用法请参见 :ref:`cn_api_guide_Name` ，一般无需设置，默认值：None。
    - **data_format** (str，可选) - 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCHW"和"NHWC"。N是批尺寸，C是通道数，H是特征高度，W是特征宽度。默认值："NCHW"。

返回：4-D Tensor，数据类型与 ``input`` 一致。如果未指定激活层，则返回卷积计算的结果，如果指定激活层，则返回卷积和激活计算之后的最终结果。

返回类型：Variable。

抛出异常：
    - ``ValueError`` - 如果 ``use_cudnn`` 不是bool值。
    - ``ValueError`` - 如果 ``data_format`` 既不是"NCHW"也不是"NHWC"。
    - ``ValueError`` - 如果 ``input`` 的通道数未被明确定义。
    - ``ValueError`` - 如果 ``padding`` 是字符串，既不是"SAME"也不是"VALID"。
    - ``ValueError`` - 如果 ``padding`` 含有4个二元组，与批尺寸对应维度的值不为0或者与通道对应维度的值不为0。
    - ``ShapeError`` - 如果输入不是4-D Tensor。
    - ``ShapeError`` - 如果输入和滤波器的维度大小不相同。
    - ``ShapeError`` - 如果输入的维度大小与 ``stride`` 之差不是2。
    - ``ShapeError`` - 如果输出的通道数不能被 ``groups`` 整除。


**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np
    data = fluid.layers.data(name='data', shape=[3, 32, 32], dtype='float32')
    param_attr = fluid.ParamAttr(name='conv2d.weight', initializer=fluid.initializer.Xavier(uniform=False), learning_rate=0.001)
    res = fluid.layers.conv2d(input=data, num_filters=2, filter_size=3, act="relu", param_attr=param_attr)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    x = np.random.rand(1, 3, 32, 32).astype("float32")
    output = exe.run(feed={"data": x}, fetch_list=[res])
    print(output)





