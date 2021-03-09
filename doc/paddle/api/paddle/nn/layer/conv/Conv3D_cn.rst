.. _cn_api_paddle_nn_Conv3D:

Conv3D
-------------------------------

.. py:class:: paddle.nn.Conv3D(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', weight_attr=None, bias_attr=None, data_format="NCDHW")



**三维卷积层**

该OP是三维卷积层（convolution3D layer），根据输入、卷积核、步长（stride）、填充（padding）、空洞大小（dilations）一组参数计算得到输出特征层大小。输入和输出是NCDHW或NDHWC格式，其中N是批尺寸，C是通道数，D是特征层深度，H是特征层高度，W是特征层宽度。三维卷积（Convlution3D）和二维卷积（Convlution2D）相似，但多了一维深度信息（depth）。如果bias_attr不为False，卷积计算会添加偏置项。

对每个输入X，有等式：

.. math::

    Out = \sigma \left ( W * X + b \right )

其中：
    - :math:`X` ：输入值，NCDHW或NDHWC格式的5-D Tensor
    - :math:`W` ：卷积核值，MCDHW格式的5-D Tensor
    - :math:`*` ：卷积操作
    - :math:`b` ：偏置值，2-D Tensor，形为 ``[M,1]``
    - :math:`\sigma` ：激活函数
    - :math:`Out` ：输出值, NCDHW或NDHWC格式的5-D Tensor，和 ``X`` 的形状可能不同

参数：
    - **in_channels** (int) - 输入图像的通道数。
    - **out_channels** (int) - 由卷积操作产生的输出的通道数。
    - **kernel_size** (int|list|tuple) - 卷积核大小。可以为单个整数或包含三个整数的元组或列表，分别表示卷积核的深度，高和宽。如果为单个整数，表示卷积核的深度，高和宽都等于该整数。
    - **stride** (int|list|tuple，可选) - 步长大小。可以为单个整数或包含三个整数的元组或列表，分别表示卷积沿着深度，高和宽的步长。如果为单个整数，表示沿着高和宽的步长都等于该整数。默认值：1。
    - **padding** (int|list|tuple|str，可选) - 填充大小。如果它是一个字符串，可以是"VALID"或者"SAME"，表示填充算法，计算细节可参考上述 ``padding`` = "SAME"或  ``padding`` = "VALID" 时的计算公式。如果它是一个元组或列表，它可以有3种格式：(1)包含5个二元组：当 ``data_format`` 为"NCDHW"时为 [[0,0], [0,0], [padding_depth_front, padding_depth_back], [padding_height_top, padding_height_bottom], [padding_width_left, padding_width_right]]，当 ``data_format`` 为"NDHWC"时为[[0,0], [padding_depth_front, padding_depth_back], [padding_height_top, padding_height_bottom], [padding_width_left, padding_width_right], [0,0]]；(2)包含6个整数值：[padding_depth_front, padding_depth_back, padding_height_top, padding_height_bottom, padding_width_left, padding_width_right]；(3)包含3个整数值：[padding_depth, padding_height, padding_width]，此时 padding_depth_front = padding_depth_back = padding_depth, padding_height_top = padding_height_bottom = padding_height, padding_width_left = padding_width_right = padding_width。若为一个整数，padding_depth = padding_height = padding_width = padding。默认值：0。
    - **dilation** (int|list|tuple，可选) - 空洞大小。可以为单个整数或包含三个整数的元组或列表，分别表示卷积核中的元素沿着深度，高和宽的空洞。如果为单个整数，表示深度，高和宽的空洞都等于该整数。默认值：1。
    - **groups** (int，可选) - 三维卷积层的组数。根据Alex Krizhevsky的深度卷积神经网络（CNN）论文中的成组卷积：当group=n，输入和卷积核分别根据通道数量平均分为n组，第一组卷积核和第一组输入进行卷积计算，第二组卷积核和第二组输入进行卷积计算，……，第n组卷积核和第n组输入进行卷积计算。默认值：1。
    - **padding_mode** (str, 可选): 填充模式。 包括 ``'zeros'``, ``'reflect'``, ``'replicate'`` 或者 ``'circular'``. 默认值: ``'zeros'`` .
    - **weight_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **bias_attr** （ParamAttr|bool，可选）- 指定偏置参数属性的对象。若 ``bias_attr`` 为bool类型，只支持为False，表示没有偏置参数。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **data_format** (str，可选) - 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCDHW"和"NDHWC"。N是批尺寸，C是通道数，D是特征深度，H是特征高度，W是特征宽度。默认值："NCDHW"。


属性
::::::::::::
.. py:attribute:: weight
本层的可学习参数，类型为 ``Parameter``

.. py:attribute:: bias
本层的可学习偏置，类型为 ``Parameter``
    
形状：

    - 输入：:math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`
    - 输出：:math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`

    其中

    .. math::

        D_{out} &= \frac{\left ( D_{in} + padding\_depth\_front + padding\_depth\_back-\left ( dilation[0]*\left ( kernel\_size[0]-1 \right )+1 \right ) \right )}{stride[0]}+1

        H_{out} &= \frac{\left ( H_{in} + padding\_height\_top + padding\_height\_bottom-\left ( dilation[1]*\left ( kernel\_size[1]-1 \right )+1 \right ) \right )}{stride[1]}+1

        W_{out} &= \frac{\left ( W_{in} + padding\_width\_left + padding\_width\_right -\left ( dilation[2]*\left ( kernel\_size[2]-1 \right )+1 \right ) \right )}{stride[2]}+1

    如果 ``padding`` = "SAME":

    .. math::
        D_{out} = \frac{(D_{in} + stride[0] - 1)}{stride[0]}

        H_{out} = \frac{(H_{in} + stride[1] - 1)}{stride[1]}

        W_{out} = \frac{(W_{in} + stride[2] - 1)}{stride[2]}

    如果 ``padding`` = "VALID":

    .. math::
        D_{out} = \frac{\left ( D_{in} -\left ( dilation[0]*\left ( kernel\_size[0]-1 \right )+1 \right ) \right )}{stride[0]}+1

        H_{out} = \frac{\left ( H_{in} -\left ( dilation[1]*\left ( kernel\_size[1]-1 \right )+1 \right ) \right )}{stride[1]}+1

        W_{out} = \frac{\left ( W_{in} -\left ( dilation[2]*\left ( kernel\_size[2]-1 \right )+1 \right ) \right )}{stride[2]}+1

抛出异常：
    - ``ValueError`` - 如果 ``data_format`` 既不是"NCDHW"也不是"NDHWC"。
    - ``ValueError`` - 如果 ``input`` 的通道数未被明确定义。
    - ``ValueError`` - 如果 ``padding`` 是字符串，既不是"SAME"也不是"VALID"。
    - ``ValueError`` - 如果 ``padding`` 含有5个二元组，与批尺寸对应维度的值不为0或者与通道对应维度的值不为0。
    - ``ShapeError`` - 如果输入不是5-D Tensor。
    - ``ShapeError`` - 如果输入和卷积核的维度大小不相同。
    - ``ShapeError`` - 如果输入的维度大小与 ``stride`` 之差不是2。
    - ``ShapeError`` - 如果输出的通道数不能被 ``groups`` 整除。


**代码示例**：

.. code-block:: python

   import paddle
   import paddle.nn as nn

   x_var = paddle.uniform((2, 4, 8, 8, 8), dtype='float32', min=-1., max=1.)

   conv = nn.Conv3D(4, 6, (3, 3, 3))
   y_var = conv(x_var)
   y_np = y_var.numpy()
   print(y_np.shape)
   # (2, 6, 6, 6, 6)

