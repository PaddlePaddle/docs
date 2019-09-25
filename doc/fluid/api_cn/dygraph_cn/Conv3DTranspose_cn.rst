.. _cn_api_fluid_dygraph_Conv3DTranspose:

Conv3DTranspose
-------------------------------

.. py:class:: paddle.fluid.dygraph.Conv3DTranspose(name_scope, num_filters, output_size=None, filter_size=None, padding=0, stride=1, dilation=1, groups=None, param_attr=None, bias_attr=None, use_cudnn=True, act=None, name=None)


3-D卷积转置层（Convlution3D transpose layer)

该层根据 输入（input）、滤波器（filter）和卷积核膨胀（dilations）、步长（stride）、填充来计算输出特征层大小或者通过output_size指定输出特征层大小。输入(Input)和输出(Output)为NCDHW格式。其中 ``N`` 为batch大小， ``C`` 为通道数（channel）, ``D``  为特征深度, ``H`` 为特征高度， ``W`` 为特征宽度。转置卷积的计算过程相当于卷积的反向计算。转置卷积又被称为反卷积（但其实并不是真正的反卷积）。欲了解卷积转置层细节，请参考下面的说明和 参考文献_ 。如果参数bias_attr不为False, 转置卷积计算会添加偏置项。如果act不为None，则转置卷积计算之后添加相应的激活函数。


.. _参考文献: http://www.matthewzeiler.com/wp-content/uploads/2017/07/cvpr2010.pdf

输入 :math:`X` 和输出 :math:`Out` 函数关系如下：

.. math::
                        \\Out=\sigma (W*X+b)\\

其中：
    -  :math:`X` : 输入图像，具有NCDHW格式的张量（Tensor）

    -  :math:`W` : 滤波器，具有NCDHW格式的张量（Tensor）

    -  :math:`*` : 卷积操作（注意：转置卷积本质上的计算还是卷积）

    -  :math:`b` : 偏置（bias），二维张量，shape为 ``[M,1]``

    -  :math:`σ` : 激活函数

    -  :math:`Out` : 输出值， ``Out`` 和 ``X`` 的 shape可能不一样


**样例**

输入:

    输入的shape：:math:`（N,C_{in}, D_{in}, H_{in}, W_{in}）`

    滤波器的shape：:math:`（C_{in}, C_{out}, D_f, H_f, W_f）`



输出:

    输出的shape：:math:`（N,C_{out}, D_{out}, H_{out}, W_{out}）`


其中：

.. math::



    D'_{out}=(D_{in}-1)*strides[0]-2*paddings[0]+dilations[0]*(D_f-1)+1
    H'_{out}=(H_{in}-1)*strides[1]-2*paddings[1]+dilations[1]*(H_f-1)+1
    W'_{out}=(W_{in}-1)*strides[2]-2*paddings[2]+dilations[2]*(W_f-1)+1
    D_{out}\in[D'_{out},D'_{out} + strides[0])
    H_{out}\in[H'_{out},H'_{out} + strides[1])
    W_{out}\in[W'_{out},W'_{out} + strides[2])

注意：
如果output_size为None，则:math:`D_{out}` = :math:`D^\prime_{out}` , :math:`H_{out}` = :math:`H^\prime_{out}` , :math:`W_{out}` = :math:`W^\prime_{out}` ;否则，指定的output_size_depth（输出特征层的深度） :math:`D_{out}` 应当介于 :math:`D^\prime_{out}` 和 :math:`D^\prime_{out} + strides[0]` 之间（不包含 :math:`D^\prime_{out} + strides[0]` ），指定的output_size_height（输出特征层的高） :math:`H_{out}` 应当介于 :math:`H^\prime_{out}` 和 :math:`H^\prime_{out} + strides[1]` 之间（不包含 :math:`H^\prime_{out} + strides[1]` ）, 并且指定的output_size_width（输出特征层的宽） :math:`W_{out}` 应当介于 :math:`W^\prime_{out}` 和 :math:`W^\prime_{out} + strides[2]` 之间（不包含 :math:`W^\prime_{out} + strides[2]` ）。 

由于转置卷积可以当成是卷积的反向计算，而根据卷积的输入输出计算公式来说，不同大小的输入特征层可能对应着相同大小的输出特征层，所以对应到转置卷积来说，固定大小的输入特征层对应的输出特征层大小并不唯一。

如果指定了output_size， ``conv3d_transpose`` 可以自动计算滤波器的大小。


参数:
      - **name_scope** （str）- 该类的名称
      - **num_filters** (int) - 滤波器（卷积核）的个数，与输出的图片的通道数相同
      - **output_size** (int|tuple，可选) - 输出图片的大小。如果 ``output_size`` 是一个元组（tuple），则该元形式为（image_H,image_W),这两个值必须为整型。如果未设置,则内部会使用filter_size、padding和stride来计算output_size。如果 ``output_size`` 和 ``filter_size`` 是同时指定的，那么它们应满足上面的公式。
      - **filter_size** (int|tuple，可选 - 滤波器大小。如果 ``filter_size`` 是一个tuple，则形式为(filter_size_H, filter_size_W)。否则，滤波器将是一个方阵。如果 ``filter_size=None`` ，则内部会计算输出大小。
      - **padding** (int|tuple) - 填充大小。如果 ``padding`` 是一个元组，它必须包含两个整数(padding_H、padding_W)。否则，padding_H = padding_W = padding。默认:padding = 0。
      - **stride** (int|tuple) - 步长大小。如果 ``stride`` 是一个元组，那么元组的形式为(stride_H、stride_W)。否则，stride_H = stride_W = stride。默认:stride = 1。
      - **dilation** (int|tuple) - 膨胀比例dilation大小。空洞卷积时会指该参数，滤波器对输入进行卷积时，感受野里每相邻两个特征点之间的空洞信息，根据`可视化效果图<https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md>`_较好理解。如果膨胀比例dilation是一个元组，那么元组的形式为(dilation_depth，dilation_height， dilation_width)。否则，dilation_depth = dilation_height = dilation_width = dilation。默认:dilation= 1。
      - **groups** (int) - 三维转置卷积层的组数。从Alex Krizhevsky的CNN Deep论文中的群卷积中受到启发，当group=2时，输入和滤波器分别根据通道数量平均分为两组，第一组滤波器和第一组输入进行卷积计算，第二组滤波器和第二组输入进行卷积计算。默认：group = 1。
      - **param_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
      - **bias_attr** (ParamAttr，可选) - 指定偏置参数属性的对象。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr`。
      - **use_cudnn** (bool) - 是否使用cudnn内核，只有已安装cudnn库时才有效。默认值为True。
      - **act** (str) -  激活函数类型，如果设置为None，则不使用激活函数。默认值为None。
      - **name** (str，可选) - 该layer的名称。如果未设置， 将自动命名该layer。默认值为True。


返回： 无

抛出异常:
    -  ``ValueError``  - 如果输入的shape、filter_size、stride、padding和groups不匹配，抛出ValueError


**代码示例**

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





