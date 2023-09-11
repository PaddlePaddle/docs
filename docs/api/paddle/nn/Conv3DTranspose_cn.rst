.. _cn_api_paddle_nn_Conv3DTranspose:

Conv3DTranspose
-------------------------------

.. py:class:: paddle.nn.Conv3DTranspose(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, dilation=1, weight_attr=None, bias_attr=None, data_format="NCDHW")


三维转置卷积层（Convlution3d transpose layer）

该层根据输入（input）、卷积核（kernel）和卷积核空洞大小（dilations）、步长（stride）、填充（padding）来计算输出特征层大小或者通过 output_size 指定输出特征层大小。输入（Input）和输出（Output）为 NCDHW 或者 NDHWC 格式。其中 N 为批尺寸，C 为通道数（channel），D 为特征深度，H 为特征层高度，W 为特征层宽度。转置卷积的计算过程相当于卷积的反向计算。转置卷积又被称为反卷积（但其实并不是真正的反卷积）。欲了解卷积转置层细节，请参考下面的说明和 `参考文献`_ 。如果参数 bias_attr 不为 False，转置卷积计算会添加偏置项。

.. _参考文献: https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf

输入 :math:`X` 和输出 :math:`Out` 函数关系如下：

.. math::
                        \\Out=\sigma (W*X+b)\\

其中：

    -  :math:`X`：输入，具有 NCDHW 或 NDHWC 格式的 5-D Tensor
    -  :math:`W`：卷积核，具有 NCDHW 格式的 5-D Tensor
    -  :math:`*`：卷积操作（**注意**：转置卷积本质上的计算还是卷积）
    -  :math:`b`：偏置（bias），1-D Tensor，形状为 ``[M]``
    -  :math:`σ`：激活函数
    -  :math:`Out`：输出值，NCDHW 或 NDHWC 格式的 5-D Tensor，和 ``X`` 的形状可能不同


.. note::
    如果 output_size 为 None，则 :math:`H_{out}` = :math:`H^\prime_{out}` , :math:`W_{out}` = :math:`W^\prime_{out}`；否则，指定的 output_size_height（输出特征层的高） :math:`H_{out}` 应当介于 :math:`H^\prime_{out}` 和 :math:`H^\prime_{out} + strides[0]` 之间（不包含 :math:`H^\prime_{out} + strides[0]` ），并且指定的 output_size_width（输出特征层的宽） :math:`W_{out}` 应当介于 :math:`W^\prime_{out}` 和 :math:`W^\prime_{out} + strides[1]` 之间（不包含 :math:`W^\prime_{out} + strides[1]` ）。

    由于转置卷积可以当成是卷积的反向计算，而根据卷积的输入输出计算公式来说，不同大小的输入特征层可能对应着相同大小的输出特征层，所以对应到转置卷积来说，固定大小的输入特征层对应的输出特征层大小并不唯一。

    如果指定了 output_size，则可以自动计算卷积核的大小。

参数
::::::::::::

  - **in_channels** (int) - 输入图像的通道数。
  - **out_channels** (int) - 卷积核的个数，和输出特征图个数相同。
  - **kernel_size** (int|list|tuple) - 卷积核大小。可以为单个整数或包含三个整数的元组或列表，分别表示卷积核的深度，高和宽。如果为单个整数，表示卷积核的深度，高和宽都等于该整数。output_size 和 kernel_size 不能同时为 None。
  - **stride** (int|tuple，可选) - 步长大小。如果 ``stride`` 为元组或列表，则必须包含三个整型数，分别表示深度，垂直和水平滑动步长。否则，表示深度，垂直和水平滑动步长均为 ``stride``。默认值为 1。
  - **padding** (int|tuple，可选) - 填充大小。如果 ``padding`` 为元组或列表，则必须包含三个整型数，分别表示深度，竖直和水平边界填充大小。否则，表示深度，竖直和水平边界填充大小均为 ``padding``。如果它是一个字符串，可以是 "VALID" 或者 "SAME" ，表示填充算法，计算细节可参考下方形状 ``padding`` = "SAME" 或  ``padding`` = "VALID" 时的计算公式。默认值为 0。
  - **output_padding** (int|list|tuple，可选) - 输出形状上一侧额外添加的大小。默认值为 0。
  - **groups** (int，可选) - 二维卷积层的组数。根据 `Alex Krizhevsky 的 Deep CNN 论文 <https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf>`_ 中的分组卷积：当 groups = 2，卷积核的前一半仅和输入特征图的前一半连接。卷积核的后一半仅和输入特征图的后一半连接。默认值为 1。
  - **dilation** (int|tuple，可选) - 空洞大小。可以为单个整数或包含三个整数的元组或列表，分别表示卷积核中的元素沿着深度，高和宽的空洞。如果为单个整数，表示深度，高和宽的空洞都等于该整数。默认值为 1。
  - **weight_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值为 None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
  - **bias_attr** (ParamAttr|bool，可选) - 指定偏置参数属性的对象。默认值为 None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
  - **data_format** (str，可选) - 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是 "NCHW" 和 "NHWC"。N 是批尺寸，C 是通道数，H 是特征高度，W 是特征宽度。默认值为 "NCDHW"。

形状
::::::::::::

    - 输入：:math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`

    - 卷积核：:math:`(C_{in}, C_{out}, K_{d}, K_{h}, K_{w})`

    - 偏置：:math:`(C_{out})`

    - 输出：:math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`

    其中

    .. math::

        D^\prime_{out} &= (D_{in} - 1) * strides[0] - 2 * paddings[0] + dilations[0] * (D_f - 1) + 1 \\
        H^\prime_{out} &= (H_{in} - 1) * strides[1] - 2 * paddings[1] + dilations[1] * (H_f - 1) + 1 \\
        W^\prime_{out} &= (W_{in} - 1) * strides[2] - 2 * paddings[2] + dilations[2] * (W_f - 1) + 1 \\
        D_{out} &\in [ D^\prime_{out}, D^\prime_{out} + strides[0] ] \\
        H_{out} &\in [ H^\prime_{out}, H^\prime_{out} + strides[1] ] \\
        W_{out} &\in [ W^\prime_{out}, W^\prime_{out} + strides[2] ] \\

    如果 ``padding`` = "SAME":

    .. math::
        & D'_{out} = \frac{(D_{in} + stride[0] - 1)}{stride[0]}\\
        & H'_{out} = \frac{(H_{in} + stride[1] - 1)}{stride[1]}\\
        & W'_{out} = \frac{(W_{in} + stride[2] - 1)}{stride[2]}\\

    如果 ``padding`` = "VALID":

    .. math::
        & D'_{out} = (D_{in}-1)*strides[0] + dilations[0]*(kernel\_size[0]-1)+1\\
        & H'_{out} = (H_{in}-1)*strides[1] + dilations[1]*(kernel\_size[1]-1)+1\\
        & W'_{out} = (W_{in}-1)*strides[2] + dilations[2]*(kernel\_size[2]-1)+1 \\

代码示例
::::::::::::

COPY-FROM: paddle.nn.Conv3DTranspose
