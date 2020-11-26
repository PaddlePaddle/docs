
conv1d_transpose
-------------------------------


.. py:function:: paddle.nn.functional.conv1d_transpose(x, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1, output_size=None, data_format='NCL', name=None)



一维转置卷积层（Convlution1D transpose layer）

该层根据输入（input）、卷积核（kernel）和空洞大小（dilations）、步长（stride）、填充（padding）来计算输出特征层大小或者通过output_size指定输出特征层大小。输入(Input)和输出(Output)为NCL或NLC格式，其中N为批尺寸，C为通道数（channel），L为特征层长度。卷积核是MCL格式，M是输出图像通道数，C是输入图像通道数，L是卷积核长度。如果组数大于1，C等于输入图像通道数除以组数的结果。转置卷积的计算过程相当于卷积的反向计算。转置卷积又被称为反卷积（但其实并不是真正的反卷积）。欲了解转置卷积层细节，请参考下面的说明和 参考文献_ 。如果参数bias_attr不为False, 转置卷积计算会添加偏置项。如果act不为None，则转置卷积计算之后添加相应的激活函数。

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

**示例**

- 输入：

    输入Tensor的形状： :math:`（N，C_{in}， L_{in}）`

    卷积核的形状 ： :math:`（C_{in}， C_{out}， L_f）`

- 输出：

    输出Tensor的形状 ： :math:`（N，C_{out}， L_{out}）`

其中

.. math::

        & L'_{out} = (L_{in}-1)*stride - padding * 2 + dilation*(L_f-1)+1\\
        & L_{out}\in[L'_{out},L'_{out} + stride)

如果 ``padding`` = "SAME":

.. math::

   L'_{out} = \frac{(L_{in} + stride - 1)}{stride}

如果 ``padding`` = "VALID":

.. math::

    L'_{out} = (L_{in}-1)*stride + dilation*(L_f-1)+1

注意：

如果output_size为None，则 :math:`L_{out}` = :math:`L^\prime_{out}` ;否则，指定的output_size（输出特征层的长度） :math:`L_{out}` 应当介于 :math:`L^\prime_{out}` 和 :math:`L^\prime_{out} + stride` 之间（不包含 :math:`L^\prime_{out} + stride` ）。

由于转置卷积可以当成是卷积的反向计算，而根据卷积的输入输出计算公式来说，不同大小的输入特征层可能对应着相同大小的输出特征层，所以对应到转置卷积来说，固定大小的输入特征层对应的输出特征层大小并不唯一。

如果指定了output_size， ``conv1d_transpose`` 可以自动计算卷积核的大小。

参数:
  - **x** (Tensor) - 输入是形状为 :math:`[N, C, L]` 或 :math:`[N, L, C]` 的3-D Tensor，N是批尺寸，C是通道数，L是特征长度，数据类型为float16, float32或float64。
  - **weight** (Tensor) - 形状为 :math:`[C, M/g, kL]` 的卷积核（卷积核）。 M是输出通道数， g是分组的个数，kL是卷积核的长度。
  - **bias** (int|list|tuple) - 偏置项，形状为： :math:`[M,]` 。
  - **stride** (int|list|tuple，可选) - 步长大小。整数或包含一个整数的列表或元组。默认值：1。
    - **padding** (int|list|tuple|str，可选) - 填充大小。可以是以下三种格式：（1）字符串，可以是"VALID"或者"SAME"，表示填充算法，计算细节可参考下述 ``padding`` = "SAME"或  ``padding`` = "VALID" 时的计算公式。（2）整数，表示在输入特征两侧各填充 ``padding`` 大小的0。（3）包含一个整数的列表或元组，表示在输入特征两侧各填充 ``padding[0]`` 大小的0. 默认值：0。
  - **output_padding** (int|list|tuple, optional): 输出形状上尾部一侧额外添加的大小. 默认值: 0.
  - **groups** (int，可选) - 一维卷积层的组数。根据Alex Krizhevsky的深度卷积神经网络（CNN）论文中的成组卷积：当group=n，输入和卷积核分别根据通道数量平均分为n组，第一组卷积核和第一组输入进行卷积计算，第二组卷积核和第二组输入进行卷积计算，……，第n组卷积核和第n组输入进行卷积计算。默认值：1。
  - **dilation** (int|list|tuple，可选) - 空洞大小。空洞卷积时会使用该参数，卷积核对输入进行卷积时，感受野里每相邻两个特征点之间的空洞信息。整数或包含一个整数的列表或元组。默认值：1。
  - **output_size** (int|list|tuple，可选) - 输出特征的长度，整数或包含一个整数的列表或元组。如果为 ``None`` , 则会用 ``filter_size``, ``padding`` 和 ``stride`` 计算出输出特征的长度。如果 ``output_size`` 和 ``filter_size`` 同时被指定，则会遵循上述公式进行计算。``output_size`` 和 ``filter_size`` 不能同时被设置为 ``None`` 。默认值：None。
  - **data_format** (str，可选) - 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCL"和"NLC"。N是批尺寸，C是通道数，L是特征长度。默认值："NCL"。
  - **name** (str，可选) – 具体用法请参见 :ref:`cn_api_guide_Name` ，一般无需设置，默认值：None。


返回：3-D Tensor，数据类型与 ``input`` 一致。如果未指定激活层，则返回转置卷积计算的结果，如果指定激活层，则返回转置卷积和激活计算之后的最终结果。

返回类型：Tensor

抛出异常:
    -  ``ValueError`` : 如果输入的shape、kernel_size、stride、padding和groups不匹配，抛出ValueError
    -  ``ValueError`` - 如果 ``data_format`` 既不是"NCL"也不是"NLC"。
    -  ``ValueError`` - 如果 ``padding`` 是字符串，既不是"SAME"也不是"VALID"。
    -  ``ValueError`` - 如果 ``output_size`` 和 ``filter_size`` 同时为None。
    -  ``ShapeError`` - 如果输入不是3-D Tensor。
    -  ``ShapeError`` - 如果输入的维度大小与 ``stride`` 之差不是2。

**代码示例**

..  code-block:: python

    import paddle
    import paddle.nn.functional as F
    import numpy as np
    
    # shape: (1, 2, 4)
    x=np.array([[[4, 0, 9, 7],
                 [8, 0, 9, 2,]]]).astype(np.float32)
    # shape: (2, 1, 2)
    w=np.array([[[7, 0]],
                [[4, 2]]]).astype(np.float32)
    x_var = paddle.to_tensor(x)
    w_var = paddle.to_tensor(w)
    y_var = F.conv1d_transpose(x_var, w_var)
    print(y_var)
    
    # [[[60. 16. 99. 75.  4.]]]
