.. _cn_api_fluid_dygraph_Conv2DTranspose:

Conv2DTranspose
-------------------------------

.. py:class:: paddle.fluid.dygraph.Conv2DTranspose(name_scope, num_filters, output_size=None, filter_size=None, padding=0, stride=1, dilation=1, groups=None, param_attr=None, bias_attr=None, use_cudnn=True, act=None)


2-D卷积转置层（Convlution2D transpose layer）

该层根据 输入（input）、滤波器（filter）和卷积核膨胀（dilations）、步长（stride）、填充（padding）来计算输出。输入(Input)和输出(Output)为NCHW格式，其中 ``N`` 为batch大小， ``C`` 为通道数（channel），``H`` 为特征高度， ``W`` 为特征宽度。参数(膨胀、步长、填充)分别都包含两个元素。这两个元素分别表示高度和宽度。欲了解卷积转置层细节，请参考下面的说明和 参考文献_ 。如果参数 ``bias_attr`` 和 ``act`` 不为 ``None``，则在卷积的输出中加入偏置，并对最终结果应用相应的激活函数。

.. _参考文献: http://www.matthewzeiler.com/wp-content/uploads/2017/07/cvpr2010.pdf

输入 :math:`X` 和输出 :math:`Out` 函数关系如下：

.. math::
                        Out=\sigma (W*X+b)\\

其中：
    -  :math:`X` : 输入张量，具有 ``NCHW`` 格式

    -  :math:`W` : 滤波器张量，具有 ``NCHW`` 格式

    -  :math:`*` : 卷积操作

    -  :math:`b` : 偏置（bias），二维张量，shape为 ``[M,1]``

    -  :math:`σ` : 激活函数

    -  :math:`Out` : 输出值，Out和 ``X`` 的 ``shape`` 可能不一样

**样例**：

输入：

.. math::

    输入张量的shape :  （N，C_{in}， H_{in}， W_{in})

    滤波器（filter）shape ： （C_{in}, C_{out}, H_f, W_f)

输出：

.. math::
    输出张量的 shape ： （N，C_{out}, H_{out}, W_{out})

其中

.. math::

        & H'_{out} = (H_{in}-1)*strides[0]-2*paddings[0]+dilations[0]*(H_f-1)+1\\
        & W'_{out} = (W_{in}-1)*strides[1]-2*paddings[1]+dilations[1]*(W_f-1)+1 \\
        & H_{out}\in[H'_{out},H'_{out} + strides[0])\\
        & W_{out}\in[W'_{out},W'_{out} + strides[1])\\



参数:
    - **name_scope** (str) - 该类的名称
    - **num_filters** (int) - 滤波器（卷积核）的个数，与输出的图片的通道数（ channel ）相同
    - **output_size** (int|tuple|None) - 输出图片的大小。如果output_size是一个元组（tuple），则该元形式为（image_H,image_W),这两个值必须为整型。如果output_size=None,则内部会使用filter_size、padding和stride来计算output_size。如果output_size和filter_size是同时指定的，那么它们应满足上面的公式。默认为None。
    - **filter_size** (int|tuple|None) - 滤波器大小。如果filter_size是一个tuple，则形式为(filter_size_H, filter_size_W)。否则，滤波器将是一个方阵。如果filter_size=None，则内部会计算输出大小。默认为None。
    - **padding** (int|tuple) - 填充大小。如果padding是一个元组，它必须包含两个整数(padding_H、padding_W)。否则，padding_H = padding_W = padding。默认:padding = 0。
    - **stride** (int|tuple) - 步长大小。如果stride是一个元组，那么元组的形式为(stride_H、stride_W)。否则，stride_H = stride_W = stride。默认:stride = 1。
    - **dilation** (int|元组) - 膨胀(dilation)大小。如果dilation是一个元组，那么元组的形式为(dilation_H, dilation_W)。否则，dilation_H = dilation_W = dilation_W。默认:dilation= 1。
    - **groups** (int) - Conv2d转置层的groups个数。从Alex Krizhevsky的CNN Deep论文中的群卷积中受到启发，当group=2时，前半部分滤波器只连接到输入通道的前半部分，而后半部分滤波器只连接到输入通道的后半部分。默认值:group = 1。
    - **param_attr** (ParamAttr|None) - conv2d_transfer中可学习参数/权重的属性。如果param_attr值为None或ParamAttr的一个属性，conv2d_transfer使用ParamAttrs作为param_attr的值。如果没有设置的param_attr初始化器，那么使用Xavier初始化。默认值:None。
    - **bias_attr** (ParamAttr|bool|None) - conv2d_tran_bias中的bias属性。如果设置为False，则不会向输出单元添加偏置。如果param_attr值为None或ParamAttr的一个属性，将conv2d_transfer使用ParamAttrs作为，bias_attr。如果没有设置bias_attr的初始化器，bias将初始化为零。默认值:None。
    - **use_cudnn** (bool) - 是否使用cudnn内核，只有已安装cudnn库时才有效。默认值:True。
    - **act** (str) -  激活函数类型，如果设置为None，则不使用激活函数。默认值:None。


返回： 存储卷积转置结果的张量。

返回类型: 变量（variable）

抛出异常:
    -  ``ValueError`` : 如果输入的shape、filter_size、stride、padding和groups不匹配，抛出ValueError

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    import numpy

    with fluid.dygraph.guard():
        data = numpy.random.random((3, 32, 32)).astype('float32')
        conv2DTranspose = fluid.dygraph.nn.Conv2DTranspose(
              'Conv2DTranspose', num_filters=2, filter_size=3)
        ret = conv2DTranspose(fluid.dygraph.base.to_variable(data))





