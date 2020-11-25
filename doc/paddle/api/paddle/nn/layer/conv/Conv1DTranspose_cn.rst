Conv1DTranspose
-------------------------------

.. py:class:: paddle.nn.Conv1DTranspose(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, dilation=1, weight_attr=None, bias_attr=None, data_format="NCL")


一维转置卷积层（Convlution1d transpose layer）

该层根据输入（input）、卷积核（kernel）和空洞大小（dilations）、步长（stride）、填充（padding）来计算输出特征大小或者通过output_size指定输出特征层大小。输入(Input)和输出(Output)为NCL或NLC格式，其中N为批尺寸，C为通道数（channel），L为特征长度。卷积核是MCL格式，M是输出图像通道数，C是输入图像通道数，L是卷积核长度。如果组数大于1，C等于输入图像通道数除以组数的结果。转置卷积的计算过程相当于卷积的反向计算。转置卷积又被称为反卷积（但其实并不是真正的反卷积）。欲了解转置卷积层细节，请参考下面的说明和 参考文献_ 。如果参数bias_attr不为False, 转置卷积计算会添加偏置项。

.. _参考文献: https://arxiv.org/pdf/1603.07285.pdf


输入 :math:`X` 和输出 :math:`Out` 函数关系如下：

.. math::
                        Out=\sigma (W*X+b)\\

其中：
    -  :math:`X` : 输入，具有NCL或NLC格式的3-D Tensor
    -  :math:`W` : 卷积核，具有NCL格式的3-D Tensor
    -  :math:`*` : 卷积计算（注意：转置卷积本质上的计算还是卷积）
    -  :math:`b` : 偏置（bias），2-D Tensor，形状为 ``[M,1]``
    -  :math:`σ` : 激活函数
    -  :math:`Out` : 输出值，NCL或NLC格式的3-D Tensor， 和 ``X`` 的形状可能不同


参数:
  - **in_channels** (int) - 输入特征的通道数。
  - **out_channels** (int) - 卷积核的个数，和输出特征通道数相同。
  - **kernel_size** (int|list|tuple) - 卷积核大小。可以为单个整数或包含一个整数的元组或列表，表示卷积核的长度。
  - **stride** (int|tuple, 可选) - 步长大小。如果 ``stride`` 为元组或列表，则必须包含一个整型数，表示滑动步长 。默认值：1。
    - **padding** (int|list|tuple|str，可选) - 填充大小。可以是以下三种格式：（1）字符串，可以是"VALID"或者"SAME"，表示填充算法，计算细节可参考下述 ``padding`` = "SAME"或  ``padding`` = "VALID" 时的计算公式。（2）整数，表示在输入特征两侧各填充 ``padding`` 大小的0。（3）包含一个整数的列表或元组，表示在输入特征两侧各填充 ``padding[0]`` 大小的0. 默认值：0。
  - **output_padding** (int|list|tuple, optional): 输出特征尾部一侧额外添加的大小. 默认值: 0.
  - **groups** (int, 可选) - 一维卷积层的组数。根据Alex Krizhevsky的深度卷积神经网络（CNN）论文中的分组卷积：当group=2，卷积核的前一半仅和输入特征图的前一半连接。卷积核的后一半仅和输入特征图的后一半连接。默认值：1。
  - **dilation** (int|tuple, 可选) - 空洞大小。可以为单个整数或包含一个整数的元组或列表，表示卷积核中的空洞。默认值：1。
  - **weight_attr** (ParamAttr, 可选) - 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
  - **bias_attr** (ParamAttr|bool, 可选) - 指定偏置参数属性的对象。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
  - **data_format** (str，可选) - 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCL"和"NLC"。N是批尺寸，C是通道数，L特征长度。默认值："NCL"。
  

形状:

    - 输入：:math:`（N，C_{in}， L_{in}）`


    - 输出：:math:`（N，C_{out}, L_{out}）`

    其中

    .. math::
        & L'_{out} = (L_{in}-1)*stride - 2*padding + dilation*(kernel\_size-1)+1\\
        & L_{out}\in[L'_{out},L'_{out} + stride)

    如果 ``padding`` = "SAME":

    .. math::
        & L'_{out} = \frac{(L_{in} + stride - 1)}{stride}

    如果 ``padding`` = "VALID":

    .. math::
        & L'_{out} = (L_{in}-1)*stride + dilation*(kernel\_size-1)+1

抛出异常:
    -  ``ValueError`` : 如果输入的shape、filter_size、stride、padding和groups不匹配，抛出ValueError
    -  ``ValueError`` - 如果 ``data_format`` 既不是"NCL"也不是"NLC"。
    -  ``ValueError`` - 如果 ``padding`` 是字符串，既不是"SAME"也不是"VALID"。
    -  ``ShapeError`` - 如果输入不是3-D Tensor。
    -  ``ShapeError`` - 如果输入和卷积核的维度大小不相同。
    -  ``ShapeError`` - 如果输入的维度大小与 ``stride`` 之差不是2。

**代码示例**

..  code-block:: python

    import paddle
    from paddle.nn import Conv1DTranspose
    import numpy as np
    
    # shape: (1, 2, 4)
    x=np.array([[[4, 0, 9, 7],
                 [8, 0, 9, 2]]]).astype(np.float32)
    # shape: (2, 1, 2)
    y=np.array([[[7, 0]],
                [[4, 2]]]).astype(np.float32)
    x_t = paddle.to_tensor(x)
    conv = Conv1DTranspose(2, 1, 2)
    conv.weight.set_value(y)
    y_t = conv(x_t)
    print(y_t)
