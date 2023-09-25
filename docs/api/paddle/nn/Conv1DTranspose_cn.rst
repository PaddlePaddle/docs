.. _cn_api_paddle_nn_Conv1DTranspose:

Conv1DTranspose
-------------------------------

.. py:class:: paddle.nn.Conv1DTranspose(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, dilation=1, weight_attr=None, bias_attr=None, data_format="NCL")


一维转置卷积层（Convlution1d transpose layer）

该层根据输入（input）、卷积核（kernel）和空洞大小（dilations）、步长（stride）、填充（padding）来计算输出特征大小或者通过 output_size 指定输出特征层大小。输入(Input)和输出(Output)为 NCL 或 NLC 格式，其中 N 为批尺寸，C 为通道数（channel），L 为特征长度。卷积核是 MCL 格式，M 是输出图像通道数，C 是输入图像通道数，L 是卷积核长度。如果组数大于 1，C 等于输入图像通道数除以组数的结果。转置卷积的计算过程相当于卷积的反向计算。转置卷积又被称为反卷积（但其实并不是真正的反卷积）。欲了解转置卷积层细节，请参考下面的说明和 `参考文献 <https://arxiv.org/pdf/1603.07285.pdf/>`_。如果参数 bias_attr 不为 False，转置卷积计算会添加偏置项。


输入 :math:`X` 和输出 :math:`Out` 函数关系如下：

.. math::
                        Out=\sigma (W*X+b)\\

其中：

    -  :math:`X`：输入，具有 NCL 或 NLC 格式的 3-D Tensor
    -  :math:`W`：卷积核，具有 NCL 格式的 3-D Tensor
    -  :math:`*`：卷积计算（注意：转置卷积本质上的计算还是卷积）
    -  :math:`b`：偏置（bias），1-D Tensor，形状为 ``[M]``
    -  :math:`σ`：激活函数
    -  :math:`Out`：输出值，NCL 或 NLC 格式的 3-D Tensor，和 ``X`` 的形状可能不同


参数
::::::::::::

  - **in_channels** (int) - 输入特征的通道数。
  - **out_channels** (int) - 卷积核的个数，和输出特征通道数相同。
  - **kernel_size** (int|list|tuple) - 卷积核大小。可以为单个整数或包含一个整数的元组或列表，表示卷积核的长度。
  - **stride** (int|tuple，可选) - 步长大小。如果 ``stride`` 为元组或列表，则必须包含一个整型数，表示滑动步长。默认值：1。
  - **padding** (int|list|tuple|str，可选) - 填充大小。可以是以下三种格式：（1）字符串，可以是"VALID"或者"SAME"，表示填充算法，计算细节可参考下述 ``padding`` = "SAME"或  ``padding`` = "VALID" 时的计算公式。（2）整数，表示在输入特征两侧各填充 ``padding`` 大小的 0。（3）包含一个整数的列表或元组，表示在输入特征两侧各填充 ``padding[0]`` 大小的 0。默认值：0。
  - **output_padding** (int|list|tuple，可选) - 输出特征尾部一侧额外添加的大小。默认值：0。
  - **groups** (int，可选) - 一维卷积层的组数。根据 Alex Krizhevsky 的深度卷积神经网络（CNN）论文中的分组卷积：当 group=2，卷积核的前一半仅和输入特征图的前一半连接。卷积核的后一半仅和输入特征图的后一半连接。默认值：1。
  - **dilation** (int|tuple，可选) - 空洞大小。可以为单个整数或包含一个整数的元组或列表，表示卷积核中的空洞。默认值：1。
  - **weight_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值为 None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
  - **bias_attr** (ParamAttr|bool，可选) - 指定偏置参数属性的对象。默认值为 None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
  - **data_format** (str，可选) - 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCL"和"NLC"。N 是批尺寸，C 是通道数，L 特征长度。默认值："NCL"。


形状
::::::::::::

    - 输入：:math:`（N，C_{in}， L_{in}）`

    - 卷积核：:math:`（C_{in}，C_{out}， K）`

    - 偏置：:math:`（C_{out}）`

    - 输出：:math:`（N，C_{out}， L_{out}）`

    其中

    .. math::
        L^\prime_{out} &= (L_{in} - 1) * stride - 2 * padding + dilation * (L_f - 1) + 1 \\
        L_{out} &\in [ L^\prime_{out}, L^\prime_{out} + stride ]

    如果 ``padding`` = "SAME":

    .. math::
        L'_{out} = \frac{(L_{in} + stride - 1)}{stride}

    如果 ``padding`` = "VALID":

    .. math::
        L'_{out} = (L_{in}-1)*stride + dilation*(L_f-1)+1

.. note::
    conv1d_transpose 可以看作 conv1d 的逆向操作。对于 conv1d ，当 stride > 1 时， conv1d 将多个输入映射到同一个输出。对于 conv1d_transpose ，当 stride > 1 时， conv1d_transpose 将同一个输入映射到多个输出。

代码示例
::::::::::::

COPY-FROM: paddle.nn.Conv1DTranspose
