.. _cn_api_nn_functional_conv1d:

conv1d
-------------------------------

.. py:function:: paddle.nn.functional.conv1d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, data_format="NCL", name=None)

该OP是一维卷积层（convolution1d layer），根据输入、卷积核、步长（stride）、填充（padding）、空洞大小（dilation）一组参数计算输出特征层大小。输入和输出是NCL或NLC格式，其中N是批尺寸，C是通道数，L是长度。卷积核是MCL格式，M是输出图像通道数，C是输入图像通道数，L是卷积核长度。如果组数(groups)大于1，C等于输入图像通道数除以组数的结果。详情请参考UFLDL's : `卷积 <http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/>`_ 。如果bias_attr不为False，卷积计算会添加偏置项。如果指定了激活函数类型，相应的激活函数会作用在最终结果上。

对每个输入X，有等式：

.. math::

    Out = \sigma \left ( W * X + b \right )

其中：
    - :math:`X` ：输入值，NCL或NLC格式的3-D Tensor
    - :math:`W` ：卷积核值，MCL格式的3-D Tensor
    - :math:`*` ：卷积操作
    - :math:`b` ：偏置值，2-D Tensor，形状为 ``[M,1]``
    - :math:`\sigma` ：激活函数
    - :math:`Out` ：输出值，NCL或NLC格式的3-D Tensor， 和 ``X`` 的形状可能不同

**示例**

- 输入：

  输入形状：:math:`（N,C_{in},L_{in}）`

  卷积核形状： :math:`（C_{out},C_{in},L_{f}）`

- 输出：

  输出形状： :math:`（N,C_{out},L_{out}）`

其中

.. math::

    L_{out} = \frac{\left ( L_{in} + padding * 2 - \left ( dilation*\left ( L_{f}-1 \right )+1 \right ) \right )}{stride}+1

如果 ``padding`` = "SAME":

.. math::
    L_{out} = \frac{(L_{in} + stride - 1)}{stride}

如果 ``padding`` = "VALID":

.. math::
    L_{out} = \frac{\left ( L_{in} -\left ( dilation*\left ( L_{f}-1 \right )+1 \right ) \right )}{stride}+1

参数：
    - **x** (Tensor) - 输入是形状为 :math:`[N, C, L]` 或 :math:`[N, L, C]` 的4-D Tensor，N是批尺寸，C是通道数，L是特征长度，数据类型为float16, float32或float64。
    - **weight** (Tensor)) - 形状为 :math:`[M, C/g, kL]` 的卷积核。 M是输出通道数， g是分组的个数，kL是卷积核的长度度。
    - **bias** (int|list|tuple，可选) - 偏置项，形状为： :math:`[M,]` 。
    - **stride** (int|list|tuple，可选) - 步长大小。卷积核和输入进行卷积计算时滑动的步长。整数或包含一个整数的列表或元组。默认值：1。
    - **padding** (int|list|tuple|str，可选) - 填充大小。可以是以下三种格式：（1）字符串，可以是"VALID"或者"SAME"，表示填充算法，计算细节可参考下述 ``padding`` = "SAME"或  ``padding`` = "VALID" 时的计算公式。（2）整数，表示在输入特征两侧各填充 ``padding`` 大小的0。（3）包含一个整数的列表或元组，表示在输入特征两侧各填充 ``padding[0]`` 大小的0. 默认值：0。
    - **dilation** (int|list|tuple，可选) - 空洞大小。空洞卷积时会使用该参数，卷积核对输入进行卷积时，感受野里每相邻两个特征点之间的空洞信息。整数或包含一个整型数的列表或元组。默认值：1。
    - **groups** (int，可选) - 一维卷积层的组数。根据Alex Krizhevsky的深度卷积神经网络（CNN）论文中的成组卷积：当group=n，输入和卷积核分别根据通道数量平均分为n组，第一组卷积核和第一组输入进行卷积计算，第二组卷积核和第二组输入进行卷积计算，……，第n组卷积核和第n组输入进行卷积计算。默认值：1。
    - **data_format** (str，可选) - 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCL"和"NLC"。N是批尺寸，C是通道数，L是特征长度。默认值："NCL"。
    - **name** (str，可选) – 具体用法请参见 :ref:`cn_api_guide_Name` ，一般无需设置，默认值：None。

返回：3-D Tensor，数据类型与 ``x`` 一致。返回卷积的结果。


**代码示例**：

.. code-block:: python

    import paddle
    import paddle.nn.functional as F
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
    
    x_var = paddle.to_tensor(x)
    w_var = paddle.to_tensor(w)
    y_var = F.conv1d(x_var, w_var)
    y_np = y_var.numpy()
    print(y_np)
    
    # [[[133. 238.]
    #   [160. 211.]]]

