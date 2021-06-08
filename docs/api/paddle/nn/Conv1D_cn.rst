.. _cn_api_paddle_nn_Conv1D:

Conv1D
-------------------------------

.. py:class:: paddle.nn.Conv1D(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', weight_attr=None, bias_attr=None, data_format="NCL")



**一维卷积层**

该OP是一维卷积层（convolution1d layer），根据输入、卷积核、步长（stride）、填充（padding）、空洞大小（dilations）一组参数计算输出特征层大小。输入和输出是NCL或NLC格式，其中N是批尺寸，C是通道数，L是特征长度。卷积核是MCL格式，M是输出特征通道数，C是输入特征通道数，L是卷积核长度度。如果组数(groups)大于1，C等于输入图像通道数除以组数的结果。详情请参考UFLDL's : `卷积 <http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/>`_ 。如果bias_attr不为False，卷积计算会添加偏置项。

对每个输入X，有等式：

.. math::

    Out = \sigma \left ( W * X + b \right )

其中：
    - :math:`X` ：输入值，NCL或NLC格式的3-D Tensor
    - :math:`W` ：卷积核值，MCL格式的3-D Tensor
    - :math:`*` ：卷积操作
    - :math:`b` ：偏置值，1-D Tensor，形状为 ``[M]``
    - :math:`\sigma` ：激活函数
    - :math:`Out` ：输出值，NCL或NLC格式的3-D Tensor， 和 ``X`` 的形状可能不同


参数：
    - **in_channels** (int) - 输入特征的通道数。
    - **out_channels** (int) - 由卷积操作产生的输出的通道数。
    - **kernel_size** (int|list|tuple) - 卷积核大小。可以为单个整数或包含一个整数的元组或列表，表示卷积核的长度。
    - **stride** (int|list|tuple，可选) - 步长大小。可以为单个整数或包含一个整数的元组或列表，表示卷积的步长。默认值：1。
    - **padding** (int|list|tuple|str，可选) - 填充大小。可以是以下三种格式：（1）字符串，可以是"VALID"或者"SAME"，表示填充算法，计算细节可参考下述 ``padding`` = "SAME"或  ``padding`` = "VALID" 时的计算公式。（2）整数，表示在输入特征两侧各填充 ``padding`` 大小的0。（3）包含一个整数的列表或元组，表示在输入特征两侧各填充 ``padding[0]`` 大小的0. 默认值：0。
    - **dilation** (int|list|tuple，可选) - 空洞大小。可以为单个整数或包含一个整数的元组或列表，表示卷积核中的元素的空洞。默认值：1。
    - **groups** (int，可选) - 一维卷积层的组数。根据Alex Krizhevsky的深度卷积神经网络（CNN）论文中的成组卷积：当group=n，输入和卷积核分别根据通道数量平均分为n组，第一组卷积核和第一组输入进行卷积计算，第二组卷积核和第二组输入进行卷积计算，……，第n组卷积核和第n组输入进行卷积计算。默认值：1。
    - **padding_mode** (str, 可选): 填充模式。 包括 ``'zeros'``, ``'reflect'``, ``'replicate'`` 或者 ``'circular'``. 默认值: ``'zeros'`` .
    - **weight_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **bias_attr** （ParamAttr|bool，可选）- 指定偏置参数属性的对象。若 ``bias_attr`` 为bool类型，只支持为False，表示没有偏置参数。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **data_format** (str，可选) - 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCL"和"NLC"。N是批尺寸，C是通道数，L是特征长度。默认值："NCL"。


属性
::::::::::::
.. py:attribute:: weight
本层的可学习参数，类型为 ``Parameter``

.. py:attribute:: bias
本层的可学习偏置，类型为 ``Parameter``
    
形状:
    - 输入: :math:`(N， C_{in}， L_{in})`
    - 卷积核: :math:`(C_{out}， C_{in}， K)`
    - 偏置: :math:`(C_{out})`
    - 输出: :math:`(N， C_{out}， L_{out})`

    其中:

    .. math::
        L_{out} = \frac{(L_{in} + 2 * padding - (dilation * (kernel\_size - 1) + 1))}{stride} + 1

    如果 ``padding`` = "SAME":

    .. math::
        L_{out} = \frac{(L_{in} + stride - 1)}{stride}

    如果 ``padding`` = "VALID":

    .. math::
        L_{out} = \frac{\left ( L_{in} -\left ( dilation*\left ( kernel\_size-1 \right )+1 \right ) \right )}{stride}+1



**代码示例**：

.. code-block:: python

   import paddle
   from paddle.nn import Conv1D
   import numpy as np
   x = np.array([[[4, 8, 1, 9],
     [7, 2, 0, 9],
     [6, 9, 2, 6]]]).astype(np.float32)
   w=np.array(
   [[[9, 3, 4],
     [0, 0, 7],
     [2, 5, 6]],
    [[0, 3, 4],
     [2, 9, 7],
     [5, 6, 8]]]).astype(np.float32)
   x_t = paddle.to_tensor(x)
   conv = Conv1D(3, 2, 3)
   conv.weight.set_value(w)
   y_t = conv(x_t)
   print(y_t)
   # [[[133. 238.]
   #   [160. 211.]]]
