.. _cn_api_paddle_nn_Conv2DTranspose:

Conv2DTranspose
-------------------------------

.. py:class:: paddle.nn.Conv2DTranspose(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, dilation=1, weight_attr=None, bias_attr=None, data_format="NCHW")


二维转置卷积层（Convlution2d transpose layer）

该层根据输入（input）、卷积核（kernel）和空洞大小（dilations）、步长（stride）、填充（padding）来计算输出特征层大小或者通过 output_size 指定输出特征层大小。输入(Input)和输出(Output)为 NCHW 或 NHWC 格式，其中 N 为批尺寸（batch size），C 为通道数（channel），H 为特征层高度，W 为特征层宽度。卷积核是 MCHW 格式，M 是输出图像通道数，C 是输入图像通道数，H 是卷积核高度，W 是卷积核宽度。如果组数大于 1，C 等于输入图像通道数除以组数的结果。转置卷积的计算过程相当于卷积的反向计算。转置卷积又被称为反卷积（但其实并不是真正的反卷积）。欲了解转置卷积层细节，请参考下面的说明和 `参考文献 <https://arxiv.org/pdf/1603.07285.pdf>`_。如果参数 bias_attr 不为 False，转置卷积计算会添加偏置项。

输入 :math:`X` 和输出 :math:`Out` 函数关系如下：

.. math::
                        Out=\sigma (W*X+b)\\

其中：

    -  :math:`X`：输入，具有 NCHW 或 NHWC 格式的 4-D Tensor
    -  :math:`W`：卷积核，具有 NCHW 格式的 4-D Tensor
    -  :math:`*`：卷积计算（注意：转置卷积本质上的计算还是卷积）
    -  :math:`b`：偏置（bias），1-D Tensor，形状为 ``[M]``
    -  :math:`σ`：激活函数
    -  :math:`Out`：输出值，NCHW 或 NHWC 格式的 4-D Tensor，和 ``X`` 的形状可能不同


注意：

如果 output_size 为 None，则 :math:`H_{out}` = :math:`H^\prime_{out}` , :math:`W_{out}` = :math:`W^\prime_{out}`；否则，指定的 output_size_height（输出特征层的高） :math:`H_{out}` 应当介于 :math:`H^\prime_{out}` 和 :math:`H^\prime_{out} + strides[0]` 之间（不包含 :math:`H^\prime_{out} + strides[0]` ），并且指定的 output_size_width（输出特征层的宽） :math:`W_{out}` 应当介于 :math:`W^\prime_{out}` 和 :math:`W^\prime_{out} + strides[1]` 之间（不包含 :math:`W^\prime_{out} + strides[1]` ）。

由于转置卷积可以当成是卷积的反向计算，而根据卷积的输入输出计算公式来说，不同大小的输入特征层可能对应着相同大小的输出特征层，所以对应到转置卷积来说，固定大小的输入特征层对应的输出特征层大小并不唯一。

如果指定了 output_size， ``conv2d_transpose`` 可以自动计算卷积核的大小。

参数
::::::::::::

  - **in_channels** (int) - 输入图像的通道数。
  - **out_channels** (int) - 卷积核的个数，和输出特征图通道数相同。
  - **kernel_size** (int|list|tuple) - 卷积核大小。可以为单个整数或包含两个整数的元组或列表，分别表示卷积核的高和宽。如果为单个整数，表示卷积核的高和宽都等于该整数。
  - **stride** (int|list|tuple，可选) - 步长大小。如果 ``stride`` 为元组或列表，则必须包含两个整型数，分别表示垂直和水平滑动步长。否则，表示垂直和水平滑动步长均为 ``stride``。默认值：1。
  - **padding** (int|str|tuple|list，可选) - 填充大小。如果 ``padding`` 为元组或列表，则必须包含两个整型数，分别表示竖直和水平边界填充大小。否则，表示竖直和水平边界填充大小均为 ``padding``。如果它是一个字符串，可以是"VALID"或者"SAME"，表示填充算法，计算细节可参考下方形状 ``padding`` = "SAME"或  ``padding`` = "VALID" 时的计算公式。默认值：0。
  - **output_padding** (int|list|tuple，可选) - 输出形状上一侧额外添加的大小。默认值：0。
  - **groups** (int，可选) - 二维卷积层的组数。根据 Alex Krizhevsky 的深度卷积神经网络（CNN）论文中的分组卷积：当 group=2，卷积核的前一半仅和输入特征图的前一半连接。卷积核的后一半仅和输入特征图的后一半连接。默认值：1。
  - **dilation** (int|list|tuple，可选) - 空洞大小。可以为单个整数或包含两个整数的元组或列表，分别表示卷积核中的元素沿着高和宽的空洞。如果为单个整数，表示高和宽的空洞都等于该整数。默认值：1。
  - **weight_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值为 None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
  - **bias_attr** (ParamAttr|bool，可选) - 指定偏置参数属性的对象。默认值为 None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
  - **data_format** (str，可选) - 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCHW"和"NHWC"。N 是批尺寸，C 是通道数，H 是特征高度，W 是特征宽度。默认值："NCHW"。


形状
::::::::::::

    - 输入：:math:`（N，C_{in}， H_{in}， W_{in}）`

    - 卷积核：:math:`(C_{in}, C_{out}, K_{h}, K_{w})`

    - 偏置：:math:`(C_{out})`

    - 输出：:math:`（N，C_{out}, H_{out}, W_{out}）`

    其中

    .. math::

        H^\prime_{out} &= (H_{in} - 1) * strides[0] - 2 * paddings[0] + dilations[0] * (H_f - 1) + 1 \\
        W^\prime_{out} &= (W_{in} - 1) * strides[1] - 2 * paddings[1] + dilations[1] * (W_f - 1) + 1 \\
        H_{out} &\in [ H^\prime_{out}, H^\prime_{out} + strides[0] ) \\
        W_{out} &\in [ W^\prime_{out}, W^\prime_{out} + strides[1] ) \\

    如果 ``padding`` = "SAME":

    .. math::
        & H'_{out} = \frac{(H_{in} + stride[0] - 1)}{stride[0]}\\
        & W'_{out} = \frac{(W_{in} + stride[1] - 1)}{stride[1]}\\

    如果 ``padding`` = "VALID":

    .. math::
        & H'_{out} = (H_{in}-1)*strides[0] + dilations[0]*(kernel\_size[0]-1)+1\\
        & W'_{out} = (W_{in}-1)*strides[1] + dilations[1]*(kernel\_size[1]-1)+1 \\


代码示例
::::::::::::

COPY-FROM: paddle.nn.Conv2DTranspose
