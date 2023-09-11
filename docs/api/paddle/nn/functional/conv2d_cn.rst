.. _cn_api_paddle_nn_functional_conv2d:

conv2d
-------------------------------

.. py:function:: paddle.nn.functional.conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, data_format="NCHW", name=None)

二维卷积层（convolution2d layer），根据输入、卷积核、步长（stride）、填充（padding）、空洞大小（dilations）一组参数计算输出特征层大小。输入和输出是 NCHW 或 NHWC 格式，其中 N 是批尺寸，C 是通道数，H 是特征高度，W 是特征宽度。卷积核是 MCHW 格式，M 是输出图像通道数，C 是输入图像通道数，H 是卷积核高度，W 是卷积核宽度。如果组数(groups)大于 1，C 等于输入图像通道数除以组数的结果。详情请参考 UFLDL's : `卷积 <http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/>`_ 。如果 bias_attr 不为 False，卷积计算会添加偏置项。如果指定了激活函数类型，相应的激活函数会作用在最终结果上。

对每个输入 X，有等式：

.. math::

    Out = \sigma \left ( W * X + b \right )

其中：

    - :math:`X`：输入值，NCHW 或 NHWC 格式的 4-D Tensor
    - :math:`W`：卷积核值，MCHW 格式的 4-D Tensor
    - :math:`*`：卷积操作
    - :math:`b`：偏置值，2-D Tensor，形状为 ``[M,1]``
    - :math:`\sigma`：激活函数
    - :math:`Out`：输出值，NCHW 或 NHWC 格式的 4-D Tensor，和 ``X`` 的形状可能不同

**示例**

- 输入：

  输入形状：:math:`（N,C_{in},H_{in},W_{in}）`

  卷积核形状：:math:`（C_{out},C_{in},H_{f},W_{f}）`

- 输出：

  输出形状：:math:`（N,C_{out},H_{out},W_{out}）`

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


参数
::::::::::::

    - **x** (Tensor) - 输入是形状为 :math:`[N, C, H, W]` 或 :math:`[N, H, W, C]` 的 4-D Tensor，N 是批尺寸，C 是通道数，H 是特征高度，W 是特征宽度，数据类型为 float16, float32 或 float64。
    - **weight** (Tensor) - 形状为 :math:`[M, C/g, kH, kW]` 的卷积核。M 是输出通道数，g 是分组的个数，kH 是卷积核的高度，kW 是卷积核的宽度。
    - **bias** (int|list|tuple，可选) - 偏置项，形状为：:math:`[M,]` 。
    - **stride** (int|list|tuple，可选) - 步长大小。卷积核和输入进行卷积计算时滑动的步长。如果它是一个列表或元组，则必须包含两个整型数：（stride_height,stride_width）。若为一个整数，stride_height = stride_width = stride。默认值：1。
    - **padding** (int|list|tuple|str，可选) - 填充大小。如果它是一个字符串，可以是"VALID"或者"SAME"，表示填充算法，计算细节可参考上述 ``padding`` = "SAME"或  ``padding`` = "VALID" 时的计算公式。如果它是一个元组或列表，它可以有 3 种格式：(1)包含 4 个二元组：当 ``data_format`` 为"NCHW"时为 [[0,0], [0,0], [padding_height_top, padding_height_bottom], [padding_width_left, padding_width_right]]，当 ``data_format`` 为"NHWC"时为[[0,0], [padding_height_top, padding_height_bottom], [padding_width_left, padding_width_right], [0,0]]；(2)包含 4 个整数值：[padding_height_top, padding_height_bottom, padding_width_left, padding_width_right]；(3)包含 2 个整数值：[padding_height, padding_width]，此时 padding_height_top = padding_height_bottom = padding_height， padding_width_left = padding_width_right = padding_width。若为一个整数，padding_height = padding_width = padding。默认值：0。
    - **dilation** (int|list|tuple，可选) - 空洞大小。空洞卷积时会使用该参数，卷积核对输入进行卷积时，感受野里每相邻两个特征点之间的空洞信息。如果空洞大小为列表或元组，则必须包含两个整型数：（dilation_height,dilation_width）。若为一个整数，dilation_height = dilation_width = dilation。默认值：1。
    - **groups** (int，可选) - 二维卷积层的组数。根据 Alex Krizhevsky 的深度卷积神经网络（CNN）论文中的成组卷积：当 group=n，输入和卷积核分别根据通道数量平均分为 n 组，第一组卷积核和第一组输入进行卷积计算，第二组卷积核和第二组输入进行卷积计算，……，第 n 组卷积核和第 n 组输入进行卷积计算。默认值：1。
    - **data_format** (str，可选) - 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCHW"和"NHWC"。N 是批尺寸，C 是通道数，H 是特征高度，W 是特征宽度。默认值："NCHW"。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
4-D Tensor，数据类型与 ``x`` 一致。返回卷积的结果。


代码示例
::::::::::::

COPY-FROM: paddle.nn.functional.conv2d
