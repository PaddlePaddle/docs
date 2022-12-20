.. _cn_api_fluid_dygraph_Conv3DTranspose:

Conv3DTranspose
-------------------------------

.. py:class:: paddle.fluid.dygraph.Conv3DTranspose(num_channels, num_filters, filter_size, output_size=None, padding=0, stride=1, dilation=1, groups=None, param_attr=None, bias_attr=None, use_cudnn=True, act=None, name=None, dtype="float32")





该接口用于构建 ``Conv3DTranspose`` 类的一个可调用对象，具体用法参照 ``代码示例`` 。3D 卷积转置层（Convlution3D transpose layer)根据输入（input）、滤波器（filter）和卷积核膨胀（dilations）、步长（stride）、填充来计算输出特征层大小或者通过 output_size 指定输出特征层大小。输入(Input)和输出(Output)为 NCDHW 格式。其中 ``N`` 为 batch 大小，``C`` 为通道数（channel）, ``D``  为特征深度，``H`` 为特征高度，``W`` 为特征宽度。转置卷积的计算过程相当于卷积的反向计算。转置卷积又被称为反卷积（但其实并不是真正的反卷积）。欲了解卷积转置层细节，请参考下面的说明和 参考文献_。如果参数 bias_attr 不为 False，转置卷积计算会添加偏置项。如果 act 不为 None，则转置卷积计算之后添加相应的激活函数。


.. _参考文献：https://arxiv.org/abs/1603.07285

输入 :math:`X` 和输出 :math:`Out` 函数关系如下：

.. math::
                        \\Out=\sigma (W*X+b)\\

其中：
    -  :math:`X`：输入图像，具有 NCDHW 格式的 Tensor

    -  :math:`W`：滤波器，具有 NCDHW 格式的 Tensor
    -  :math:`*`：卷积操作（注意：转置卷积本质上的计算还是卷积）

    -  :math:`b`：偏置(bias)，维度为 :math:`[M,1]` 的 2D Tensor

    -  :math:`σ`：激活函数

    -  :math:`Out`：输出值，``Out`` 和 ``X`` 的 shape 可能不一样


**样例**

输入：

    输入 Tensor 的维度：:math:`[N,C_{in}, D_{in}, H_{in}, W_{in}]`

    滤波器 Tensor 的维度：:math:`[C_{in}, C_{out}, D_f, H_f, W_f]`



输出：

    输出 Tensor 的维度：:math:`[N,C_{out}, D_{out}, H_{out}, W_{out}]`


其中：

.. math::
    D'_{out}=(D_{in}-1)*strides[0]-2*paddings[0]+dilations[0]*(D_f-1)+1 \\
    H'_{out}=(H_{in}-1)*strides[1]-2*paddings[1]+dilations[1]*(H_f-1)+1 \\
    W'_{out}=(W_{in}-1)*strides[2]-2*paddings[2]+dilations[2]*(W_f-1)+1 \\
.. math::
    D_{out}\in[D'_{out},D'_{out} + strides[0]) \\
    H_{out}\in[H'_{out},H'_{out} + strides[1]) \\
    W_{out}\in[W'_{out},W'_{out} + strides[2])


**注意** :
    如果 output_size 为 None，则 :math:`D_{out}` = :math:`D^\prime_{out}` , :math:`H_{out}` = :math:`H^\prime_{out}` , :math:`W_{out}` = :math:`W^\prime_{out}`；否则，指定的 output_size_depth（输出特征层的深度） :math:`D_{out}` 应当介于 :math:`D^\prime_{out}` 和 :math:`D^\prime_{out} + strides[0]` 之间（不包含 :math:`D^\prime_{out} + strides[0]` ），指定的 output_size_height（输出特征层的高） :math:`H_{out}` 应当介于 :math:`H^\prime_{out}` 和 :math:`H^\prime_{out} + strides[1]` 之间（不包含 :math:`H^\prime_{out} + strides[1]` ），并且指定的 output_size_width（输出特征层的宽） :math:`W_{out}` 应当介于 :math:`W^\prime_{out}` 和 :math:`W^\prime_{out} + strides[2]` 之间（不包含 :math:`W^\prime_{out} + strides[2]` ）。

    由于转置卷积可以当成是卷积的反向计算，而根据卷积的输入输出计算公式来说，不同大小的输入特征层可能对应着相同大小的输出特征层，所以对应到转置卷积来说，固定大小的输入特征层对应的输出特征层大小并不唯一。

    如果指定了 output_size，其可以自动计算滤波器的大小。


参数
::::::::::::

      - **num_channels** (int) - 输入图像的通道数。
      - **num_filters** (int) - 滤波器（卷积核）的个数，与输出的图片的通道数相同。
      - **filter_size** (int|tuple) - 滤波器大小。如果 filter_size 是一个元组，则必须包含三个整型数，（filter_size_depth，filter_size_height, filter_size_width）。否则，filter_size_depth = filter_size_height = filter_size_width = filter_size。如果 filter_size=None，则必须指定 output_size，其会根据 output_size、padding 和 stride 计算出滤波器大小。
      - **output_size** (int|tuple，可选) - 输出图片的大小。如果 ``output_size`` 是一个元组（tuple），则该元形式为（image_H,image_W)，这两个值必须为整型。如果未设置，则内部会使用 filter_size、padding 和 stride 来计算 output_size。如果 ``output_size`` 和 ``filter_size`` 是同时指定的，那么它们应满足上面的公式。默认值为 None。output_size 和 filter_size 不能同时为 None。
      - **padding** (int|tuple，可选) - 填充 padding 大小。padding 参数在输入特征层每边添加 ``dilation * (kernel_size - 1) - padding`` 个 0。如果 padding 是一个元组，它必须包含三个整数(padding_depth，padding_height，padding_width)。否则，padding_depth = padding_height = padding_width = padding。默认值为 0。
      - **stride** (int|tuple，可选) - 步长 stride 大小。滤波器和输入进行卷积计算时滑动的步长。如果 stride 是一个元组，那么元组的形式为(stride_depth，stride_height，stride_width)。否则，stride_depth = stride_height = stride_width = stride。默认值为 1。
      - **dilation** (int|tuple，可选) - 膨胀比例 dilation 大小。空洞卷积时会指该参数，滤波器对输入进行卷积时，感受野里每相邻两个特征点之间的空洞信息，根据  `可视化效果图  <https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md>`_ 较好理解。如果膨胀比例 dilation 是一个元组，那么元组的形式为(dilation_depth，dilation_height， dilation_width)。否则，dilation_depth = dilation_height = dilation_width = dilation。默认值为 1。
      - **groups** (int，可选) - 三维转置卷积层的组数。从 `Alex Krizhevsky 的 Deep CNN 论文 <https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf>`_ 中的群卷积中受到启发，当 group=2 时，输入和滤波器分别根据通道数量平均分为两组，第一组滤波器和第一组输入进行卷积计算，第二组滤波器和第二组输入进行卷积计算。默认值为 1。
      - **param_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值为 None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
      - **bias_attr** (ParamAttr，可选) - 指定偏置参数属性的对象。默认值为 None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
      - **use_cudnn** (bool，可选) - 是否使用 cudnn 内核，只有安装 Paddle GPU 版时才有效。默认值为 True。
      - **act** (str，可选) -  激活函数类型，如果设置为 None，则不使用激活函数。默认值为 None。
      - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
      - **dtype** (str，可选) - 数据类型，可以为"float32"或"float64"。默认值为"float32"。


返回
::::::::::::
 无

代码示例
::::::::::::

..  code-block:: python

    import paddle.fluid as fluid
    import numpy

    with fluid.dygraph.guard():
        data = numpy.random.random((5, 3, 12, 32, 32)).astype('float32')

        conv3dTranspose = fluid.dygraph.nn.Conv3DTranspose(
               'Conv3DTranspose',
               num_filters=12,
               filter_size=12,
               use_cudnn=False)
        ret = conv3dTranspose(fluid.dygraph.base.to_variable(data))

属性
::::::::::::
属性
::::::::::::
weight
'''''''''

本层的可学习参数，类型为 ``Parameter``

bias
'''''''''

本层的可学习偏置，类型为 ``Parameter``
