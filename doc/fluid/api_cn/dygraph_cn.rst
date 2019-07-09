###################
fluid.dygraph
###################



.. _cn_api_fluid_dygraph_BackwardStrategy:

BackwardStrategy
-------------------------------

.. py:class:: paddle.fluid.dygraph.BackwardStrategy

BackwardStrategy是描述反向过程的描述符，现有如下功能:

1. ``sort_sum_gradient`` 按回溯逆序将梯度加和


**代码示例**

.. code-block:: python

    import numpy as np
    import paddle.fluid as fluid
    from paddle.fluid import FC

    x = np.ones([2, 2], np.float32)
    with fluid.dygraph.guard():
        inputs2 = []
        for _ in range(10):
            inputs2.append(fluid.dygraph.base.to_variable(x))
        ret2 = fluid.layers.sums(inputs2)
        loss2 = fluid.layers.reduce_sum(ret2)
        backward_strategy = fluid.dygraph.BackwardStrategy()
        backward_strategy.sort_sum_gradient = True
        loss2.backward(backward_strategy)




.. _cn_api_fluid_dygraph_BatchNorm:

BatchNorm
-------------------------------

.. py:class:: paddle.fluid.dygraph.BatchNorm(name_scope, num_channels, act=None, is_test=False, momentum=0.9, epsilon=1e-05, param_attr=None, bias_attr=None, dtype='float32', data_layout='NCHW', in_place=False, moving_mean_name=None, moving_variance_name=None, do_model_average_for_mean_and_var=False, fuse_with_relu=False, use_global_stats=False, trainable_statistics=False)


批正则化层（Batch Normalization Layer）

可用作conv2d和全连接操作的正则化函数。该层需要的数据格式如下：

1.NHWC[batch,in_height,in_width,in_channels]
2.NCHW[batch,in_channels,in_height,in_width]

更多详情请参考 : `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift <https://arxiv.org/pdf/1502.03167.pdf>`_

``input`` 是mini-batch的输入特征。

.. math::
    \mu_{\beta}        &\gets \frac{1}{m} \sum_{i=1}^{m} x_i                                 \quad &// mini-batch-mean \\
    \sigma_{\beta}^{2} &\gets \frac{1}{m} \sum_{i=1}^{m}(x_i - \mu_{\beta})^2               \quad &// mini-batch-variance \\
    \hat{x_i}          &\gets \frac{x_i - \mu_\beta} {\sqrt{\sigma_{\beta}^{2} + \epsilon}}  \quad &// normalize \\
    y_i &\gets \gamma \hat{x_i} + \beta                                                      \quad &// scale-and-shift

当use_global_stats = True时， :math:`\mu_{\beta}` 和 :math:`\sigma_{\beta}^{2}` 不是一个minibatch的统计数据。 它们是全局（或运行）统计数据。 （它通常来自预训练模型）。训练和测试（或预测）具有相同的行为：

.. math::

    \hat{x_i} &\gets \frac{x_i - \mu_\beta} {\sqrt{\
    \sigma_{\beta}^{2} + \epsilon}}  \\
    y_i &\gets \gamma \hat{x_i} + \beta



参数：
    - **name_scope** (str) - 该类的名称
    - **act** （string，默认None）- 激活函数类型，linear|relu|prelu|...
    - **is_test** （bool,默认False） - 指示它是否在测试阶段。
    - **momentum** （float，默认0.9）- 此值用于计算 moving_mean and moving_var. 更新公式为:  :math:`moving\_mean = moving\_mean * momentum + new\_mean * (1. - momentum` :math:`moving\_var = moving\_var * momentum + new\_var * (1. - momentum` ， 默认值0.9.
    - **epsilon** （float，默认1e-05）- 加在分母上为了数值稳定的值。默认值为1e-5。
    - **param_attr** （ParamAttr|None） - batch_norm参数范围的属性，如果设为None或者是ParamAttr的一个属性，batch_norm创建ParamAttr为param_attr。如果没有设置param_attr的初始化函数，参数初始化为Xavier。默认：None
    - **bias_attr** （ParamAttr|None） - batch_norm bias参数的属性，如果设为None或者是ParamAttr的一个属性，batch_norm创建ParamAttr为bias_attr。如果没有设置bias_attr的初始化函数，参数初始化为0。默认：None
    - **data_layout** （string,默认NCHW) - NCHW|NHWC。默认NCHW
    - **in_place** （bool，默认False）- 得出batch norm可复用记忆的输入和输出
    - **moving_mean_name** （string，默认None）- moving_mean的名称，存储全局Mean均值。 
    - **moving_variance_name** （string，默认None）- moving_variance的名称，存储全局方差。 
    - **do_model_average_for_mean_and_var** （bool，默认False）- 是否为mean和variance做模型均值
    - **fuse_with_relu** （bool）- 如果为True，batch norm后该操作符执行relu。默认为False。
    - **use_global_stats** （bool, Default False） – 是否使用全局均值和方差。 在预测或测试模式下，将use_global_stats设置为true或将is_test设置为true，并且行为是等效的。 在训练模式中，当设置use_global_stats为True时，在训练期间也使用全局均值和方差。
    - **trainable_statistics** （bool）- eval模式下是否计算mean均值和var方差。eval模式下，trainable_statistics为True时，由该批数据计算均值和方差。默认为False。

返回： 张量，在输入中运用批正则后的结果

返回类型：变量（Variable）

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid

    with fluid.dygraph.guard():
        fc = fluid.FC('fc', size=200, param_attr='fc1.w')
        hidden1 = fc(x)
        batch_norm = fluid.BatchNorm("batch_norm", 10)
        hidden2 = batch_norm(hidden1)


.. py:method:: add_parameter(name, parameter)

添加参数实例。被添加的参数实例的访问方式和self.name类似。

参数：
    - **name** - 该子层的命名
    - **parameter** - Parameter实例

返回：   传入的参数实例    


.. py:method:: add_sublayer(name, sublayer)

添加子层实例。被添加的子层实例的访问方式和self.name类似。

参数：
    - **name** - 该子层的命名
    - **sublayer** - Layer实例

返回：   传入的子层



.. py:method:: create_parameter(attr, shape, dtype, is_bias=False, default_initializer=None)

为层(layers)创建参数。

参数：
    - **attr** (ParamAttr)- 参数的参数属性
    - **shape** - 参数的形状
    - **dtype** - 参数的数据类型
    - **is_bias** - 是否为偏置bias参数      
    - **default_initializer** - 设置参数的默认初始化方法

返回：    创建的参数变量



.. py:method:: create_variable(name=None, persistable=None, dtype=None, type=VarType.LOD_TENSOR)

为层创建变量

参数：
    - **name** - 变量名
    - **persistable** - 是否为持久性变量
    - **dtype** - 变量中的数据类型
    - **type** - 变量类型   

返回： 创建的变量(Variable)

.. py:method:: full_name()

层的全名。

组成方式如下：

name_scope + “/” + MyLayer.__class__.__name__

返回：  层的全名


.. py:method:: parameters(include_sublayers=True)

返回一个由当前层及其子层的参数组成的列表。

参数：
    - **include_sublayers** - 如果为True，返回的列表中包含子层的参数

返回：  一个由当前层及其子层的参数组成的列表


.. py:method:: sublayers(include_sublayers=True)

返回一个由所有子层组成的列表。

参数：
    - **include_sublayers** - 如果为True，则包括子层中的各层

返回： 一个由所有子层组成的列表





.. _cn_api_fluid_dygraph_BilinearTensorProduct:

BilinearTensorProduct
-------------------------------

.. py:class:: paddle.fluid.dygraph.BilinearTensorProduct(name_scope, size, name=None, act=None, param_attr=None, bias_attr=None)

该层可将一对张量进行双线性乘积计算，例如：

.. math::

    out_{i} = x * W_{i} * {y^\mathrm{T}}, i=0,1,...,size-1

式中，

- :math:`x` ： 第一个输入，分别包含M个元素，形为[batch_size, M]
- :math:`y` ：第二个输入，分别包含N个元素，形为[batch_size, N]
- :math:`W_i` ：第i个学习到的权重，形为[M,N]
- :math:`out_i` ：输出的第i个元素
- :math:`y^T` ： :math:`y_2` 的转置


参数：
    - **name_scope**  (str) – 类的名称。
    - **size**  (int) – 该层的维度大小。
    - **act**  (str) – 对输出应用的激励函数。默认:None。
    - **name**  (str) – 该层的名称。 默认: None。
    - **param_attr**  (ParamAttr) – 该层中可学习权重/参数w的参数属性。默认: None.
    - **bias_attr**  (ParamAttr) – 该层中偏置(bias)的参数属性。若为False, 则输出中不应用偏置。如果为None, 偏置默认为0。默认: None.

返回：形为 [batch_size, size]的二维张量

返回类型： Variable

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    import numpy

    with fluid.dygraph.guard():
        layer1 = numpy.random.random((5, 5)).astype('float32')
        layer2 = numpy.random.random((5, 4)).astype('float32')
        bilinearTensorProduct = fluid.dygraph.nn.BilinearTensorProduct(
               'BilinearTensorProduct', size=1000)
        ret = bilinearTensorProduct(fluid.dygraph.base.to_variable(layer1),
                           fluid.dygraph.base.to_variable(layer2))


.. py:method:: add_parameter(name, parameter)

添加参数实例。被添加的参数实例的访问方式和self.name类似。

参数：
    - **name** - 该子层的命名
    - **parameter** - Parameter实例

返回：   传入的参数实例    


.. py:method:: add_sublayer(name, sublayer)

添加子层实例。被添加的子层实例的访问方式和self.name类似。

参数：
    - **name** - 该子层的命名
    - **sublayer** - Layer实例

返回：   传入的子层



.. py:method:: create_parameter(attr, shape, dtype, is_bias=False, default_initializer=None)

为层(layers)创建参数。

参数：
    - **attr** (ParamAttr)- 参数的参数属性
    - **shape** - 参数的形状
    - **dtype** - 参数的数据类型
    - **is_bias** - 是否为偏置bias参数      
    - **default_initializer** - 设置参数的默认初始化方法

返回：    创建的参数变量



.. py:method:: create_variable(name=None, persistable=None, dtype=None, type=VarType.LOD_TENSOR)

为层创建变量

参数：
    - **name** - 变量名
    - **persistable** - 是否为持久性变量
    - **dtype** - 变量中的数据类型
    - **type** - 变量类型   

返回： 创建的变量(Variable)

.. py:method:: full_name()

层的全名。

组成方式如下：

name_scope + “/” + MyLayer.__class__.__name__

返回：  层的全名


.. py:method:: parameters(include_sublayers=True)

返回一个由当前层及其子层的参数组成的列表。

参数：
    - **include_sublayers** - 如果为True，返回的列表中包含子层的参数

返回：  一个由当前层及其子层的参数组成的列表


.. py:method:: sublayers(include_sublayers=True)

返回一个由所有子层组成的列表。

参数：
    - **include_sublayers** - 如果为True，则包括子层中的各层

返回： 一个由所有子层组成的列表





.. _cn_api_fluid_dygraph_Conv2D:

Conv2D
-------------------------------

.. py:class:: paddle.fluid.dygraph.Conv2D(name_scope, num_filters, filter_size, stride=1, padding=0, dilation=1, groups=None, param_attr=None, bias_attr=None, use_cudnn=True, act=None, dtype='float32')

卷积二维层（convolution2D layer）根据输入、滤波器（filter）、步长（stride）、填充（padding）、dilations、一组参数计算输出。输入和输出是NCHW格式，N是批尺寸，C是通道数，H是特征高度，W是特征宽度。滤波器是MCHW格式，M是输出图像通道数，C是输入图像通道数，H是滤波器高度，W是滤波器宽度。如果组数大于1，C等于输入图像通道数除以组数的结果。详情请参考UFLDL's : `卷积 <http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/>`_ 。如果提供了bias属性和激活函数类型，bias会添加到卷积（convolution）的结果中相应的激活函数会作用在最终结果上。

对每个输入X，有等式：

.. math::

    Out = \sigma \left ( W * X + b \right )

其中：
    - :math:`X` ：输入值，NCHW格式的张量（Tensor）
    - :math:`W` ：滤波器值，MCHW格式的张量（Tensor）
    - :math:`*` ： 卷积操作
    - :math:`b` ：Bias值，二维张量（Tensor），shape为 ``[M,1]``
    - :math:`\sigma` ：激活函数
    - :math:`Out` ：输出值，``Out`` 和 ``X`` 的shape可能不同

**示例**

- 输入：

  输入shape：:math:`( N,C_{in},H_{in},W_{in} )`

  滤波器shape： :math:`( C_{out},C_{in},H_{f},W_{f} )`

- 输出：

  输出shape： :math:`( N,C_{out},H_{out},W_{out} )`

其中

.. math::

    H_{out} = \frac{\left ( H_{in}+2*paddings[0]-\left ( dilations[0]*\left ( H_{f}-1 \right )+1 \right ) \right )}{strides[0]}+1

    W_{out} = \frac{\left ( W_{in}+2*paddings[1]-\left ( dilations[1]*\left ( W_{f}-1 \right )+1 \right ) \right )}{strides[1]}+1

参数：
    - **name_scope** (str) - 该类的名称
    - **num_fliters** (int) - 滤波器数。和输出图像通道相同
    - **filter_size** (int|tuple|None) - 滤波器大小。如果filter_size是一个元组，则必须包含两个整型数，（filter_size，filter_size_W）。否则，滤波器为square
    - **stride** (int|tuple) - 步长(stride)大小。如果步长（stride）为元组，则必须包含两个整型数，（stride_H,stride_W）。否则，stride_H = stride_W = stride。默认：stride = 1
    - **padding** (int|tuple) - 填充（padding）大小。如果填充（padding）为元组，则必须包含两个整型数，（padding_H,padding_W)。否则，padding_H = padding_W = padding。默认：padding = 0
    - **dilation** (int|tuple) - 膨胀（dilation）大小。如果膨胀（dialation）为元组，则必须包含两个整型数，（dilation_H,dilation_W）。否则，dilation_H = dilation_W = dilation。默认：dilation = 1
    - **groups** (int) - 卷积二维层（Conv2D Layer）的组数。根据Alex Krizhevsky的深度卷积神经网络（CNN）论文中的成组卷积：当group=2，滤波器的前一半仅和输入通道的前一半连接。滤波器的后一半仅和输入通道的后一半连接。默认：groups = 1
    - **param_attr** (ParamAttr|None) - conv2d的可学习参数/权重的参数属性。如果设为None或者ParamAttr的一个属性，conv2d创建ParamAttr为param_attr。如果param_attr的初始化函数未设置，参数则初始化为 :math:`Normal(0.0,std)` ，并且std为 :math:`\frac{2.0}{filter\_elem\_num}^{0.5}` 。默认为None
    - **bias_attr** (ParamAttr|bool|None) - conv2d bias的参数属性。如果设为False，则没有bias加到输出。如果设为None或者ParamAttr的一个属性，conv2d创建ParamAttr为bias_attr。如果bias_attr的初始化函数未设置，bias初始化为0.默认为None
    - **use_cudnn** （bool） - 是否用cudnn核，仅当下载cudnn库才有效。默认：True
    - **act** (str) - 激活函数类型，如果设为None，则未添加激活函数。默认：None


抛出异常:
  - ``ValueError`` - 如果输入shape和filter_size，stride,padding和groups不匹配。


**代码示例**

.. code-block:: python

    from paddle.fluid.dygraph.base import to_variable
    import paddle.fluid as fluid
    from paddle.fluid.dygraph import Conv2D
    import numpy as np

    data = np.random.uniform( -1, 1, [10, 3, 32, 32] ).astype('float32')
    with fluid.dygraph.guard():
        conv2d = Conv2D( "conv2d", 2, 3)
        data = to_variable( data )
        conv = conv2d( data )




.. py:method:: add_parameter(name, parameter)

添加参数实例。被添加的参数实例的访问方式和self.name类似。

参数：
    - **name** - 该子层的命名
    - **parameter** - Parameter实例

返回：   传入的参数实例    


.. py:method:: add_sublayer(name, sublayer)

添加子层实例。被添加的子层实例的访问方式和self.name类似。

参数：
    - **name** - 该子层的命名
    - **sublayer** - Layer实例

返回：   传入的子层



.. py:method:: create_parameter(attr, shape, dtype, is_bias=False, default_initializer=None)

为层(layers)创建参数。

参数：
    - **attr** (ParamAttr)- 参数的参数属性
    - **shape** - 参数的形状
    - **dtype** - 参数的数据类型
    - **is_bias** - 是否为偏置bias参数      
    - **default_initializer** - 设置参数的默认初始化方法

返回：    创建的参数变量



.. py:method:: create_variable(name=None, persistable=None, dtype=None, type=VarType.LOD_TENSOR)

为层创建变量

参数：
    - **name** - 变量名
    - **persistable** - 是否为持久性变量
    - **dtype** - 变量中的数据类型
    - **type** - 变量类型   

返回： 创建的变量(Variable)

.. py:method:: full_name()

层的全名。

组成方式如下：

name_scope + “/” + MyLayer.__class__.__name__

返回：  层的全名


.. py:method:: parameters(include_sublayers=True)

返回一个由当前层及其子层的参数组成的列表。

参数：
    - **include_sublayers** - 如果为True，返回的列表中包含子层的参数

返回：  一个由当前层及其子层的参数组成的列表


.. py:method:: sublayers(include_sublayers=True)

返回一个由所有子层组成的列表。

参数：
    - **include_sublayers** - 如果为True，则包括子层中的各层

返回： 一个由所有子层组成的列表





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


.. py:method:: add_parameter(name, parameter)

添加参数实例。被添加的参数实例的访问方式和self.name类似。

参数：
    - **name** - 该子层的命名
    - **parameter** - Parameter实例

返回：   传入的参数实例    


.. py:method:: add_sublayer(name, sublayer)

添加子层实例。被添加的子层实例的访问方式和self.name类似。

参数：
    - **name** - 该子层的命名
    - **sublayer** - Layer实例

返回：   传入的子层



.. py:method:: create_parameter(attr, shape, dtype, is_bias=False, default_initializer=None)

为层(layers)创建参数。

参数：
    - **attr** (ParamAttr)- 参数的参数属性
    - **shape** - 参数的形状
    - **dtype** - 参数的数据类型
    - **is_bias** - 是否为偏置bias参数      
    - **default_initializer** - 设置参数的默认初始化方法

返回：    创建的参数变量



.. py:method:: create_variable(name=None, persistable=None, dtype=None, type=VarType.LOD_TENSOR)

为层创建变量

参数：
    - **name** - 变量名
    - **persistable** - 是否为持久性变量
    - **dtype** - 变量中的数据类型
    - **type** - 变量类型   

返回： 创建的变量(Variable)

.. py:method:: full_name()

层的全名。

组成方式如下：

name_scope + “/” + MyLayer.__class__.__name__

返回：  层的全名


.. py:method:: parameters(include_sublayers=True)

返回一个由当前层及其子层的参数组成的列表。

参数：
    - **include_sublayers** - 如果为True，返回的列表中包含子层的参数

返回：  一个由当前层及其子层的参数组成的列表


.. py:method:: sublayers(include_sublayers=True)

返回一个由所有子层组成的列表。

参数：
    - **include_sublayers** - 如果为True，则包括子层中的各层

返回： 一个由所有子层组成的列表


.. _cn_api_fluid_dygraph_Conv3D:

Conv3D
-------------------------------

.. py:class:: paddle.fluid.dygraph.Conv3D(name_scope, num_filters, filter_size, stride=1, padding=0, dilation=1, groups=None, param_attr=None, bias_attr=None, use_cudnn=True, act=None)


3D卷积层（convolution3D layer）根据输入、滤波器（filter）、步长（stride）、填充（padding）、膨胀（dilations）、组数参数计算得到输出。输入和输出是NCHW格式，N是批尺寸，C是通道数，H是特征高度，W是特征宽度。卷积三维（Convlution3D）和卷积二维（Convlution2D）相似，但多了一维深度（depth）。如果提供了bias属性和激活函数类型，bias会添加到卷积（convolution）的结果中相应的激活函数会作用在最终结果上。

对每个输入X，有等式：

.. math::


    Out = \sigma \left ( W * X + b \right )

其中：
    - :math:`X` ：输入值，NCDHW格式的张量（Tensor）
    - :math:`W` ：滤波器值，MCDHW格式的张量（Tensor）
    - :math:`*` ： 卷积操作
    - :math:`b` ：Bias值，二维张量（Tensor），形为 ``[M,1]``
    - :math:`\sigma` ：激活函数
    - :math:`Out` ：输出值, 和 ``X`` 的形状可能不同

**示例**

- 输入：
    输入shape： :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`

    滤波器shape： :math:`(C_{out}, C_{in}, D_f, H_f, W_f)`
- 输出：
    输出shape： :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`

其中

.. math::


    D_{out}&= \frac{(D_{in} + 2 * paddings[0] - (dilations[0] * (D_f - 1) + 1))}{strides[0]} + 1 \\
    H_{out}&= \frac{(H_{in} + 2 * paddings[1] - (dilations[1] * (H_f - 1) + 1))}{strides[1]} + 1 \\
    W_{out}&= \frac{(W_{in} + 2 * paddings[2] - (dilations[2] * (W_f - 1) + 1))}{strides[2]} + 1

参数：
    - **name_scope** (str) - 该类的名称
    - **num_fliters** (int) - 滤波器数。和输出图像通道相同
    - **filter_size** (int|tuple|None) - 滤波器大小。如果filter_size是一个元组，则必须包含三个整型数，(filter_size_D, filter_size_H, filter_size_W)。否则，滤波器为棱长为int的立方体形。
    - **stride** (int|tuple) - 步长(stride)大小。如果步长（stride）为元组，则必须包含三个整型数， (stride_D, stride_H, stride_W)。否则，stride_D = stride_H = stride_W = stride。默认：stride = 1
    - **padding** (int|tuple) - 填充（padding）大小。如果填充（padding）为元组，则必须包含三个整型数，(padding_D, padding_H, padding_W)。否则， padding_D = padding_H = padding_W = padding。默认：padding = 0
    - **dilation** (int|tuple) - 膨胀（dilation）大小。如果膨胀（dialation）为元组，则必须包含两个整型数， (dilation_D, dilation_H, dilation_W)。否则，dilation_D = dilation_H = dilation_W = dilation。默认：dilation = 1
    - **groups** (int) - 卷积二维层（Conv2D Layer）的组数。根据Alex Krizhevsky的深度卷积神经网络（CNN）论文中的成组卷积：当group=2，滤波器的前一半仅和输入通道的前一半连接。滤波器的后一半仅和输入通道的后一半连接。默认：groups = 1
    - **param_attr** (ParamAttr|None) - conv2d的可学习参数/权重的参数属性。如果设为None或者ParamAttr的一个属性，conv2d创建ParamAttr为param_attr。如果param_attr的初始化函数未设置，参数则初始化为 :math:`Normal(0.0,std)`，并且std为 :math:`\left ( \frac{2.0}{filter\_elem\_num} \right )^{0.5}` 。默认为None
    - **bias_attr** (ParamAttr|bool|None) - conv2d bias的参数属性。如果设为False，则没有bias加到输出。如果设为None或者ParamAttr的一个属性，conv2d创建ParamAttr为bias_attr。如果bias_attr的初始化函数未设置，bias初始化为0.默认为None
    - **use_cudnn** （bool） - 是否用cudnn核，仅当下载cudnn库才有效。默认：True
    - **act** (str) - 激活函数类型，如果设为None，则未添加激活函数。默认：None


返回：张量，存储卷积和非线性激活结果

返回类型：变量（Variable）

抛出异常：
  - ``ValueError`` - 如果 ``input`` 的形和 ``filter_size`` ， ``stride`` , ``padding`` 和 ``groups`` 不匹配。

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import numpy

    with fluid.dygraph.guard():
        data = numpy.random.random((5, 3, 12, 32, 32)).astype('float32')
        conv3d = fluid.dygraph.nn.Conv3D(
              'Conv3D', num_filters=2, filter_size=3, act="relu")
        ret = conv3d(fluid.dygraph.base.to_variable(data))



.. py:method:: add_parameter(name, parameter)

添加参数实例。被添加的参数实例的访问方式和self.name类似。

参数：
    - **name** - 该子层的命名
    - **parameter** - Parameter实例

返回：   传入的参数实例    


.. py:method:: add_sublayer(name, sublayer)

添加子层实例。被添加的子层实例的访问方式和self.name类似。

参数：
    - **name** - 该子层的命名
    - **sublayer** - Layer实例

返回：   传入的子层



.. py:method:: create_parameter(attr, shape, dtype, is_bias=False, default_initializer=None)

为层(layers)创建参数。

参数：
    - **attr** (ParamAttr)- 参数的参数属性
    - **shape** - 参数的形状
    - **dtype** - 参数的数据类型
    - **is_bias** - 是否为偏置bias参数      
    - **default_initializer** - 设置参数的默认初始化方法

返回：    创建的参数变量



.. py:method:: create_variable(name=None, persistable=None, dtype=None, type=VarType.LOD_TENSOR)

为层创建变量

参数：
    - **name** - 变量名
    - **persistable** - 是否为持久性变量
    - **dtype** - 变量中的数据类型
    - **type** - 变量类型   

返回： 创建的变量(Variable)

.. py:method:: full_name()

层的全名。

组成方式如下：

name_scope + “/” + MyLayer.__class__.__name__

返回：  层的全名


.. py:method:: parameters(include_sublayers=True)

返回一个由当前层及其子层的参数组成的列表。

参数：
    - **include_sublayers** - 如果为True，返回的列表中包含子层的参数

返回：  一个由当前层及其子层的参数组成的列表


.. py:method:: sublayers(include_sublayers=True)

返回一个由所有子层组成的列表。

参数：
    - **include_sublayers** - 如果为True，则包括子层中的各层

返回： 一个由所有子层组成的列表



.. _cn_api_fluid_dygraph_Conv3DTranspose:

Conv3DTranspose
-------------------------------

.. py:class:: paddle.fluid.dygraph.Conv3DTranspose(name_scope, num_filters, output_size=None, filter_size=None, padding=0, stride=1, dilation=1, groups=None, param_attr=None, bias_attr=None, use_cudnn=True, act=None, name=None)


3-D卷积转置层（Convlution3D transpose layer)

该层根据 输入（input）、滤波器（filter）和卷积核膨胀（dilations）、步长（stride）、填充来计算输出。输入(Input)和输出(Output)为NCDHW格式。其中 ``N`` 为batch大小， ``C`` 为通道数（channel）, ``D``  为特征深度, ``H`` 为特征高度， ``W`` 为特征宽度。参数(膨胀、步长、填充)分别包含两个元素。这两个元素分别表示高度和宽度。欲了解卷积转置层细节，请参考下面的说明和 参考文献_ 。如果参数 ``bias_attr`` 和 ``act`` 不为None，则在卷积的输出中加入偏置，并对最终结果应用相应的激活函数

.. _参考文献: http://www.matthewzeiler.com/wp-content/uploads/2017/07/cvpr2010.pdf

输入X和输出Out函数关系X，有等式如下：

.. math::
                        \\Out=\sigma (W*X+b)\\

其中：
    -  :math:`X` : 输入张量，具有 ``NCDHW`` 格式

    -  :math:`W` : 滤波器张量，，具有 ``NCDHW`` 格式

    -  :math:`*` : 卷积操作

    -  :math:`b` : 偏置（bias），二维张量，shape为 ``[M,1]``

    -  :math:`σ` : 激活函数

    -  :math:`Out` : 输出值， ``Out`` 和 ``X`` 的 shape可能不一样


**样例**

输入:

    输入形状: :math:`(N,C_{in},D_{in},H_{in},W_{in})` 

    Filter形状: :math:`(C_{in},C_{out},D_f,H_f,W_f)` 



输出:

    输出形状: :math:`(N,C_{out},D_{out},H_{out},W_{out})`


其中：

.. math::



    D_{out}=(D_{in}-1)*strides[0]-2*paddings[0]+dilations[0]*(D_f-1)+1

    H_{out}=(H_{in}-1)*strides[1]-2*paddings[1]+dilations[1]*(H_f-1)+1

    W_{out}=(W_{in}-1)*strides[2]-2*paddings[2]+dilations[2]*(W_f-1)+1



参数:
      - **name_scope** （str）- 该类的名称
      - **num_filters** (int) - 滤波器（卷积核）的个数，与输出的图片的通道数（channel）相同
      - **output_size** (int|tuple|None) - 输出图片的大小。如果 ``output_size`` 是一个元组（tuple），则该元形式为（image_H,image_W),这两个值必须为整型。如果 ``output_size=None`` ,则内部会使用filter_size、padding和stride来计算output_size。如果 ``output_size`` 和 ``filter_size`` 是同时指定的，那么它们应满足上面的公式。
      - **filter_size** (int|tuple|None) - 滤波器大小。如果 ``filter_size`` 是一个tuple，则形式为(filter_size_H, filter_size_W)。否则，滤波器将是一个方阵。如果 ``filter_size=None`` ，则内部会计算输出大小。
      - **padding** (int|tuple) - 填充大小。如果 ``padding`` 是一个元组，它必须包含两个整数(padding_H、padding_W)。否则，padding_H = padding_W = padding。默认:padding = 0。
      - **stride** (int|tuple) - 步长大小。如果 ``stride`` 是一个元组，那么元组的形式为(stride_H、stride_W)。否则，stride_H = stride_W = stride。默认:stride = 1。
      - **dilation** (int|元组) - 膨胀大小。如果 ``dilation`` 是一个元组，那么元组的形式为(dilation_H, dilation_W)。否则，dilation_H = dilation_W = dilation_W。默认:dilation= 1。
      - **groups** (int) - Conv2d转置层的groups个数。从Alex Krizhevsky的CNN Deep论文中的群卷积中受到启发，当group=2时，前半部分滤波器只连接到输入通道的前半部分，而后半部分滤波器只连接到输入通道的后半部分。默认值:group = 1。
      - **param_attr** (ParamAttr|None) - conv2d_transfer中可学习参数/权重的属性。如果param_attr值为None或ParamAttr的一个属性，conv2d_transfer使用ParamAttrs作为param_attr的值。如果没有设置的param_attr初始化器，那么使用Xavier初始化。默认值:None。
      - **bias_attr** (ParamAttr|bool|None) - conv2d_tran_bias中的bias属性。如果设置为False，则不会向输出单元添加偏置。如果param_attr值为None或ParamAttr的一个属性，将conv2d_transfer使用ParamAttrs作为，bias_attr。如果没有设置bias_attr的初始化器，bias将初始化为零。默认值:None。
      - **use_cudnn** (bool) - 是否使用cudnn内核，只有已安装cudnn库时才有效。默认值:True。
      - **act** (str) -  激活函数类型，如果设置为None，则不使用激活函数。默认值:None。
      - **name** (str|None) - 该layer的名称(可选)。如果设置为None， 将自动命名该layer。默认值:True。


返回： 存储卷积转置结果的张量。

返回类型: 变量（variable）

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


.. py:method:: add_parameter(name, parameter)

添加参数实例。被添加的参数实例的访问方式和self.name类似。

参数：
    - **name** - 该子层的命名
    - **parameter** - Parameter实例

返回：   传入的参数实例    


.. py:method:: add_sublayer(name, sublayer)

添加子层实例。被添加的子层实例的访问方式和self.name类似。

参数：
    - **name** - 该子层的命名
    - **sublayer** - Layer实例

返回：   传入的子层



.. py:method:: create_parameter(attr, shape, dtype, is_bias=False, default_initializer=None)

为层(layers)创建参数。

参数：
    - **attr** (ParamAttr)- 参数的参数属性
    - **shape** - 参数的形状
    - **dtype** - 参数的数据类型
    - **is_bias** - 是否为偏置bias参数      
    - **default_initializer** - 设置参数的默认初始化方法

返回：    创建的参数变量



.. py:method:: create_variable(name=None, persistable=None, dtype=None, type=VarType.LOD_TENSOR)

为层创建变量

参数：
    - **name** - 变量名
    - **persistable** - 是否为持久性变量
    - **dtype** - 变量中的数据类型
    - **type** - 变量类型   

返回： 创建的变量(Variable)

.. py:method:: full_name()

层的全名。

组成方式如下：

name_scope + “/” + MyLayer.__class__.__name__

返回：  层的全名


.. py:method:: parameters(include_sublayers=True)

返回一个由当前层及其子层的参数组成的列表。

参数：
    - **include_sublayers** - 如果为True，返回的列表中包含子层的参数

返回：  一个由当前层及其子层的参数组成的列表


.. py:method:: sublayers(include_sublayers=True)

返回一个由所有子层组成的列表。

参数：
    - **include_sublayers** - 如果为True，则包括子层中的各层

返回： 一个由所有子层组成的列表



.. _cn_api_fluid_dygraph_CosineDecay:

CosineDecay
-------------------------------

.. py:class:: paddle.fluid.dygraph.CosineDecay(learning_rate, step_each_epoch, epochs, begin=0, step=1, dtype='float32')

使用 cosine decay 的衰减方式进行学习率调整。

在训练模型时，建议一边进行训练一边降低学习率。 通过使用此方法，学习率将通过如下cosine衰减策略进行衰减：

.. math::

    decayed\_lr = learning\_rate * 0.5 * (math.cos * (epoch * \frac{math.pi}{epochs} ) + 1)


参数：
    - **learning_rate** (Variable | float) - 初始学习率。
    - **step_each_epoch** （int） - 一次迭代中的步数。
    - **begin** (int) - 起始步，默认为0。
    - **step** (int) - 步大小，默认为1。
    - **dtype**  (str) - 学习率的dtype，默认为‘float32’


**代码示例**

.. code-block:: python

    base_lr = 0.1
    with fluid.dygraph.guard():
        optimizer  = fluid.optimizer.SGD(
            learning_rate = fluid.dygraph.CosineDecay(
                    base_lr, 10000, 120) )


.. py:method:: create_lr_var(lr)

将学习率(lr)从float类型转为Variable类型。

参数：
    - **lr** - 学习率

返回：Variable类型的学习率


.. _cn_api_fluid_dygraph_Embedding:

Embedding
-------------------------------

.. py:class:: paddle.fluid.dygraph.Embedding(name_scope, size, is_sparse=False, is_distributed=False, padding_idx=None, param_attr=None, dtype='float32')

Embedding层

该层用于在查找表中查找 ``input`` 中的ID对应的embeddings。查找的结果是input里每个ID对应的embedding。
所有的输入变量都作为局部变量传入LayerHelper构造器

参数：
    - **name_scope** (str)-该类的名称。
    - **size** (tuple|list)-查找表参数的维度。应当有两个参数，一个代表嵌入矩阵字典的大小，一个代表每个嵌入向量的大小。
    - **is_sparse** (bool)-代表是否用稀疏更新的标志。
    - **is_distributed** (bool)-是否从远程参数服务端运行查找表。
    - **padding_idx** (int|long|None)-如果为 ``None`` ，对查找结果无影响。如果 ``padding_idx`` 不为空，表示一旦查找表中找到input中对应的 ``padding_idx``，则用0填充输出结果。如果 ``padding_idx`` <0 ,则在查找表中使用的 ``padding_idx`` 值为 :math:`size[0]+dim` 。默认：None。
    - **param_attr** (ParamAttr)-该层参数。默认为None。
    - **dtype** (np.dtype|core.VarDesc.VarType|str)-数据类型：float32,float_16,int等。默认:‘float32’

返回：张量，存储已有输入的嵌入矩阵。

返回类型：变量(Variable)

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.dygraph.base as base
    import numpy as np

    inp_word = np.array([[[1]]]).astype('int64')
    dict_size = 20
    with fluid.dygraph.guard():
        emb = fluid.dygraph.Embedding(
            name_scope='embedding',
            size=[dict_size, 32],
            param_attr='emb.w',
            is_sparse=False)
        static_rlt3 = emb(base.to_variable(inp_word))

.. py:method:: add_parameter(name, parameter)

添加参数实例。被添加的参数实例的访问方式和self.name类似。

参数：
    - **name** - 该子层的命名
    - **parameter** - Parameter实例

返回：   传入的参数实例    


.. py:method:: add_sublayer(name, sublayer)

添加子层实例。被添加的子层实例的访问方式和self.name类似。

参数：
    - **name** - 该子层的命名
    - **sublayer** - Layer实例

返回：   传入的子层



.. py:method:: create_parameter(attr, shape, dtype, is_bias=False, default_initializer=None)

为层(layers)创建参数。

参数：
    - **attr** (ParamAttr)- 参数的参数属性
    - **shape** - 参数的形状
    - **dtype** - 参数的数据类型
    - **is_bias** - 是否为偏置bias参数      
    - **default_initializer** - 设置参数的默认初始化方法

返回：    创建的参数变量



.. py:method:: create_variable(name=None, persistable=None, dtype=None, type=VarType.LOD_TENSOR)

为层创建变量

参数：
    - **name** - 变量名
    - **persistable** - 是否为持久性变量
    - **dtype** - 变量中的数据类型
    - **type** - 变量类型   

返回： 创建的变量(Variable)

.. py:method:: full_name()

层的全名。

组成方式如下：

name_scope + “/” + MyLayer.__class__.__name__

返回：  层的全名


.. py:method:: parameters(include_sublayers=True)

返回一个由当前层及其子层的参数组成的列表。

参数：
    - **include_sublayers** - 如果为True，返回的列表中包含子层的参数

返回：  一个由当前层及其子层的参数组成的列表


.. py:method:: sublayers(include_sublayers=True)

返回一个由所有子层组成的列表。

参数：
    - **include_sublayers** - 如果为True，则包括子层中的各层

返回： 一个由所有子层组成的列表





.. _cn_api_fluid_dygraph_enabled:

enabled
-------------------------------

.. py:function:: paddle.fluid.dygraph.enabled()

文档正在施工中，不日将为您呈现！



.. _cn_api_fluid_dygraph_ExponentialDecay:

ExponentialDecay
-------------------------------

.. py:class:: paddle.fluid.dygraph.ExponentialDecay(learning_rate, decay_steps, decay_rate, staircase=False, begin=0, step=1, dtype='float32')

对学习率应用指数衰减。

在学习率上运用指数衰减。
训练模型时，推荐在训练过程中降低学习率。每次 ``decay_steps`` 步骤中用 ``decay_rate`` 衰减学习率。

.. code-block:: text

    if staircase == True:
        decayed_learning_rate = learning_rate * decay_rate ^ floor(global_step / decay_steps)
    else:
        decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)

参数：
    - **learning_rate** (Variable|float)-初始学习率
    - **decay_steps** (int)-见以上衰减运算
    - **decay_rate** (float)-衰减率。见以上衰减运算
    - **staircase** (Boolean)-若为True,按离散区间衰减学习率。默认：False
    - **begin** (int) - 起始步，默认为0。
    - **step** (int) - 步大小，默认为1。
    - **dtype**  (str) - 学习率的dtype，默认为‘float32’


**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    base_lr = 0.1
    with fluid.dygraph.guard():
        sgd_optimizer = fluid.optimizer.SGD(
              learning_rate=fluid.dygraph.ExponentialDecay(
                  learning_rate=base_lr,
                  decay_steps=10000,
                  decay_rate=0.5,
                  staircase=True))


.. py:method:: create_lr_var(lr)

将学习率(lr)从float类型转为Variable类型。

参数：
    - **lr** - 学习率

返回：Variable类型的学习率




.. _cn_api_fluid_dygraph_FC:

FC
-------------------------------

.. py:class:: paddle.fluid.dygraph.FC(name_scope, size, num_flatten_dims=1, param_attr=None, bias_attr=None, act=None, is_test=False, dtype='float32')




**全连接层**

该函数在神经网络中建立一个全连接层。 它可以将一个或多个tensor（ ``input`` 可以是一个list或者Variable，详见参数说明）作为自己的输入，并为每个输入的tensor创立一个变量，称为“权”（weights），等价于一个从每个输入单元到每个输出单元的全连接权矩阵。FC层用每个tensor和它对应的权相乘得到形状为[M, size]输出tensor，M是批大小。如果有多个输入tensor，那么形状为[M, size]的多个输出张量的结果将会被加起来。如果 ``bias_attr`` 非空，则会新创建一个偏向变量（bias variable），并把它加入到输出结果的运算中。最后，如果 ``act`` 非空，它也会加入最终输出的计算中。

当输入为单个张量：

.. math::

        \\Out = Act({XW + b})\\



当输入为多个张量：

.. math::

        \\Out=Act(\sum^{N-1}_{i=0}X_iW_i+b) \\


上述等式中：
  - :math:`N` ：输入的数目,如果输入是变量列表，N等于len（input）
  - :math:`X_i` ：第i个输入的tensor
  - :math:`W_i` ：对应第i个输入张量的第i个权重矩阵
  - :math:`b` ：该层创立的bias参数
  - :math:`Act` ：activation function(激励函数)
  - :math:`Out` ：输出tensor

::

            Given:
                data_1.data = [[[0.1, 0.2],
                               [0.3, 0.4]]]
                data_1.shape = (1, 2, 2) # 1 is batch_size

                data_2 = [[[0.1, 0.2, 0.3]]]
                data_2.shape = (1, 1, 3)

                out = fluid.layers.fc(input=[data_1, data_2], size=2)

            Then:
                out.data = [[0.18669507, 0.1893476]]
                out.shape = (1, 2)


参数:
  - **name_scope** (str) – 该类的名称
  - **size** (int) – 该层输出单元的数目
  - **num_flatten_dims** (int, 默认为1) – fc层可以接受一个维度大于2的tensor。此时， 它首先会被扁平化(flattened)为一个二维矩阵。 参数 ``num_flatten_dims`` 决定了输入tensor的flattened方式: 前 ``num_flatten_dims`` (包含边界，从1开始数) 个维度会被扁平化为最终矩阵的第一维 (维度即为矩阵的高), 剩下的 rank(X) - num_flatten_dims 维被扁平化为最终矩阵的第二维 (即矩阵的宽)。 例如， 假设X是一个五维tensor，其形可描述为(2, 3, 4, 5, 6), 且num_flatten_dims = 3。那么扁平化的矩阵形状将会如此： (2 x 3 x 4, 5 x 6) = (24, 30)
  - **param_attr** (ParamAttr|list of ParamAttr|None) – 该层可学习的参数/权的参数属性
  - **bias_attr** (ParamAttr|list of ParamAttr, default None) – 该层bias变量的参数属性。如果值为False， 则bias变量不参与输出单元运算。 如果值为None，bias变量被初始化为0。默认为 None。
  - **act** (str|None) – 应用于输出的Activation（激励函数）
  - **is_test** (bool) – 表明当前执行是否处于测试阶段的标志
  - **dtype** (str) – 权重的数据类型


弹出异常：``ValueError`` - 如果输入tensor的维度小于2

**代码示例**

..  code-block:: python

    from paddle.fluid.dygraph.base import to_variable
    import paddle.fluid as fluid
    from paddle.fluid.dygraph import FC
    import numpy as np

    data = np.random.uniform( -1, 1, [30, 10, 32] ).astype('float32')
    with fluid.dygraph.guard():
        fc = FC( "fc", 64, num_flatten_dims=2)
        data = to_variable( data )
        conv = fc( data )

.. py:method:: add_parameter(name, parameter)

添加参数实例。被添加的参数实例的访问方式和self.name类似。

参数：
    - **name** - 该子层的命名
    - **parameter** - Parameter实例

返回：   传入的参数实例    


.. py:method:: add_sublayer(name, sublayer)

添加子层实例。被添加的子层实例的访问方式和self.name类似。

参数：
    - **name** - 该子层的命名
    - **sublayer** - Layer实例

返回：   传入的子层



.. py:method:: create_parameter(attr, shape, dtype, is_bias=False, default_initializer=None)

为层(layers)创建参数。

参数：
    - **attr** (ParamAttr)- 参数的参数属性
    - **shape** - 参数的形状
    - **dtype** - 参数的数据类型
    - **is_bias** - 是否为偏置bias参数      
    - **default_initializer** - 设置参数的默认初始化方法

返回：    创建的参数变量



.. py:method:: create_variable(name=None, persistable=None, dtype=None, type=VarType.LOD_TENSOR)

为层创建变量

参数：
    - **name** - 变量名
    - **persistable** - 是否为持久性变量
    - **dtype** - 变量中的数据类型
    - **type** - 变量类型   

返回： 创建的变量(Variable)

.. py:method:: full_name()

层的全名。

组成方式如下：

name_scope + “/” + MyLayer.__class__.__name__

返回：  层的全名


.. py:method:: parameters(include_sublayers=True)

返回一个由当前层及其子层的参数组成的列表。

参数：
    - **include_sublayers** - 如果为True，返回的列表中包含子层的参数

返回：  一个由当前层及其子层的参数组成的列表


.. py:method:: sublayers(include_sublayers=True)

返回一个由所有子层组成的列表。

参数：
    - **include_sublayers** - 如果为True，则包括子层中的各层

返回： 一个由所有子层组成的列表


.. _cn_api_fluid_dygraph_GroupNorm:

GroupNorm
-------------------------------

.. py:class:: paddle.fluid.dygraph.GroupNorm(name_scope, groups, epsilon=1e-05, param_attr=None, bias_attr=None, act=None, data_layout='NCHW')

**Group Normalization层**

请参考 `Group Normalization <https://arxiv.org/abs/1803.08494>`_ 。

参数：
    - **name_scope** (str) - 该类名称
    - **groups** (int) - 从 channel 中分离出来的 group 的数目
    - **epsilon** (float) - 为防止方差除零，增加一个很小的值
    - **param_attr** (ParamAttr|None)  - 可学习标度的参数属性 :math:`g`,如果设置为False，则不会向输出单元添加标度。如果设置为0，偏差初始化为1。默认值:None
    - **bias_attr** (ParamAttr|None) - 可学习偏置的参数属性 :math:`b ` , 如果设置为False，则不会向输出单元添加偏置量。如果设置为零，偏置初始化为零。默认值:None。
    - **act** (str) - 将激活应用于输出的 group normalizaiton
    - **data_layout** (string|NCHW) - 只支持NCHW。

返回： 一个张量变量，它是对输入进行 group normalization 后的结果。

返回类型：Variable


**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    import numpy

    with fluid.dygraph.guard():
        x = numpy.random.random((8, 32, 32)).astype('float32')
        groupNorm = fluid.dygraph.nn.GroupNorm('GroupNorm', groups=4)
        ret = groupNorm(fluid.dygraph.base.to_variable(x))


.. py:method:: add_parameter(name, parameter)

添加参数实例。被添加的参数实例的访问方式和self.name类似。

参数：
    - **name** - 该子层的命名
    - **parameter** - Parameter实例

返回：   传入的参数实例    


.. py:method:: add_sublayer(name, sublayer)

添加子层实例。被添加的子层实例的访问方式和self.name类似。

参数：
    - **name** - 该子层的命名
    - **sublayer** - Layer实例

返回：   传入的子层



.. py:method:: create_parameter(attr, shape, dtype, is_bias=False, default_initializer=None)

为层(layers)创建参数。

参数：
    - **attr** (ParamAttr)- 参数的参数属性
    - **shape** - 参数的形状
    - **dtype** - 参数的数据类型
    - **is_bias** - 是否为偏置bias参数      
    - **default_initializer** - 设置参数的默认初始化方法

返回：    创建的参数变量



.. py:method:: create_variable(name=None, persistable=None, dtype=None, type=VarType.LOD_TENSOR)

为层创建变量

参数：
    - **name** - 变量名
    - **persistable** - 是否为持久性变量
    - **dtype** - 变量中的数据类型
    - **type** - 变量类型   

返回： 创建的变量(Variable)

.. py:method:: full_name()

层的全名。

组成方式如下：

name_scope + “/” + MyLayer.__class__.__name__

返回：  层的全名


.. py:method:: parameters(include_sublayers=True)

返回一个由当前层及其子层的参数组成的列表。

参数：
    - **include_sublayers** - 如果为True，返回的列表中包含子层的参数

返回：  一个由当前层及其子层的参数组成的列表


.. py:method:: sublayers(include_sublayers=True)

返回一个由所有子层组成的列表。

参数：
    - **include_sublayers** - 如果为True，则包括子层中的各层

返回： 一个由所有子层组成的列表



.. _cn_api_fluid_dygraph_GRUUnit:

GRUUnit
-------------------------------

.. py:class:: paddle.fluid.dygraph.GRUUnit(name_scope, size, param_attr=None, bias_attr=None, activation='tanh', gate_activation='sigmoid', origin_mode=False, dtype='float32')

GRU单元层。GRU执行步骤基于如下等式：


如果origin_mode为True，则该运算公式来自论文
`Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling  <https://arxiv.org/pdf/1412.3555.pdf>`_ 。

公式如下:

.. math::
    u_t=actGate(xu_t+W_{u}h_{t-1}+b_u)
.. math::
    r_t=actGate(xr_t+W_{r}h_{t-1}+b_r)
.. math::
    m_t=actNode(xm_t+W_{c}dot(r_t,h_{t-1})+b_m)
.. math::
    h_t=dot((1-u_t),m_t)+dot(u_t,h_{t-1})


如果origin_mode为False，则该运算公式来自论文
`Learning Phrase Representations using RNN Encoder Decoder for Statistical Machine Translation <https://arxiv.org/pdf/1406.1078.pdf>`_ 。

.. math::
    u_t & = act_g(W_{ux}x_{t} + W_{uh}h_{t-1} + b_u)\\
    r_t & = act_g(W_{rx}x_{t} + W_{rh}h_{t-1} + b_r)\\
    \tilde{h_t} & = act_c(W_{cx}x_{t} + W_{ch}(r_t \odot h_{t-1}) + b_c)\\
    h_t & = u_t \odot h_{t-1} + (1-u_t) \odot \tilde{h_t}


GRU单元的输入包括 :math:`z_t` ， :math:`h_{t-1}` 。在上述等式中， :math:`z_t` 会被分割成三部分： :math:`xu_t` 、 :math:`xr_t` 和 :math:`xm_t`  。
这意味着要为一批输入实现一个全GRU层，我们需要采用一个全连接层，才能得到 :math:`z_t=W_{fc}x_t` 。
:math:`u_t` 和 :math:`r_t` 分别代表了GRU神经元的update gates（更新门）和reset gates(重置门)。
和LSTM不同，GRU少了一个门（它没有LSTM的forget gate）。但是它有一个叫做中间候选隐藏状态（intermediate candidate hidden output）的输出，
记为 :math:`m_t` 。 该层有三个输出： :math:`h_t, dot(r_t,h_{t-1})` 以及 :math:`u_t，r_t，m_t` 的连结(concatenation)。




参数:
    - **name_scope** (str) – 该类的名称
    - **size** (int) – 输入数据的维度
    - **param_attr** (ParamAttr|None) – 可学习的隐藏层权重矩阵的参数属性。
    注意：
      - 该权重矩阵形为 :math:`(T×3D)` ， :math:`D` 是隐藏状态的规模（hidden size）
      - 该权重矩阵的所有元素由两部分组成， 一是update gate和reset gate的权重，形为 :math:`(D×2D)` ；二是候选隐藏状态（candidate hidden state）的权重矩阵，形为 :math:`(D×D)`
      如果该函数参数值为None或者 ``ParamAttr`` 类中的属性之一，gru_unit则会创建一个 ``ParamAttr`` 类的对象作为param_attr。如果param_attr没有被初始化，那么会由Xavier来初始化它。默认值为None
    - **bias_attr** (ParamAttr|bool|None) - GRU的bias变量的参数属性。形为 :math:`(1x3D)` 的bias连结（concatenate）在update gates（更新门），reset gates(重置门)以及candidate calculations（候选隐藏状态计算）中的bias。如果值为False，那么上述三者将没有bias参与运算。若值为None或者 ``ParamAttr`` 类中的属性之一，gru_unit则会创建一个 ``ParamAttr`` 类的对象作为 bias_attr。如果bias_attr没有被初始化，那它会被默认初始化为0。默认值为None。
    - **activation** (str) –  神经元 “actNode” 的激励函数（activation）类型。默认类型为‘tanh’
    - **gate_activation** (str) – 门 “actGate” 的激励函数（activation）类型。 默认类型为 ‘sigmoid’。
    - **dtype** (str) – 该层的数据类型，默认为‘float32’。


返回：  hidden value（隐藏状态的值），reset-hidden value(重置隐藏状态值)，gate values(门值)

返回类型:  元组（tuple）

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.dygraph.base as base
    import numpy

    lod = [[2, 4, 3]]
    D = 5
    T = sum(lod[0])

    hidden_input = numpy.random.rand(T, D).astype('float32')
    with fluid.dygraph.guard():
        x = numpy.random.random((3, 32, 32)).astype('float32')
        gru = fluid.dygraph.GRUUnit('gru', size=D * 3)
        dy_ret = gru(
          base.to_variable(input), base.to_variable(hidden_input))



.. py:method:: add_parameter(name, parameter)

添加参数实例。被添加的参数实例的访问方式和self.name类似。

参数：
    - **name** - 该子层的命名
    - **parameter** - Parameter实例

返回：   传入的参数实例    


.. py:method:: add_sublayer(name, sublayer)

添加子层实例。被添加的子层实例的访问方式和self.name类似。

参数：
    - **name** - 该子层的命名
    - **sublayer** - Layer实例

返回：   传入的子层



.. py:method:: create_parameter(attr, shape, dtype, is_bias=False, default_initializer=None)

为层(layers)创建参数。

参数：
    - **attr** (ParamAttr)- 参数的参数属性
    - **shape** - 参数的形状
    - **dtype** - 参数的数据类型
    - **is_bias** - 是否为偏置bias参数      
    - **default_initializer** - 设置参数的默认初始化方法

返回：    创建的参数变量



.. py:method:: create_variable(name=None, persistable=None, dtype=None, type=VarType.LOD_TENSOR)

为层创建变量

参数：
    - **name** - 变量名
    - **persistable** - 是否为持久性变量
    - **dtype** - 变量中的数据类型
    - **type** - 变量类型   

返回： 创建的变量(Variable)

.. py:method:: full_name()

层的全名。

组成方式如下：

name_scope + “/” + MyLayer.__class__.__name__

返回：  层的全名


.. py:method:: parameters(include_sublayers=True)

返回一个由当前层及其子层的参数组成的列表。

参数：
    - **include_sublayers** - 如果为True，返回的列表中包含子层的参数

返回：  一个由当前层及其子层的参数组成的列表


.. py:method:: sublayers(include_sublayers=True)

返回一个由所有子层组成的列表。

参数：
    - **include_sublayers** - 如果为True，则包括子层中的各层

返回： 一个由所有子层组成的列表



.. _cn_api_fluid_dygraph_guard:

guard
-------------------------------

.. py:function:: paddle.fluid.dygraph.guard(place=None)

创建一个dygraph上下文，用于运行dygraph。

参数：
    - **place** (fluid.CPUPlace|fluid.CUDAPlace|None) – 执行场所

返回： None

**代码示例**

.. code-block:: python

    import numpy as np
    import paddle.fluid as fluid

    with fluid.dygraph.guard():
        inp = np.ones([3, 32, 32], dtype='float32')
        t = fluid.dygraph.base.to_variable(inp)
        fc1 = fluid.FC('fc1', size=4, bias_attr=False, num_flatten_dims=1)
        fc2 = fluid.FC('fc2', size=4)
        ret = fc1(t)
        dy_ret = fc2(ret)





