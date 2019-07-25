.. _cn_api_fluid_nets_simple_img_conv_pool:

simple_img_conv_pool
-------------------------------

.. py:function:: paddle.fluid.nets.simple_img_conv_pool(input, num_filters, filter_size, pool_size, pool_stride, pool_padding=0, pool_type='max', global_pooling=False, conv_stride=1, conv_padding=0, conv_dilation=1, conv_groups=1, param_attr=None, bias_attr=None, act=None, use_cudnn=True)

 ``simple_img_conv_pool`` 由一个Convolution2d和一个Pool2d组成。

参数：
    - **input** （Variable） - 输入图像的格式为[N，C，H，W]。
    - **num_filters** （int） - ``filter`` 的数量。它与输出的通道相同。
    - **filter_size** （int | list | tuple） - 过滤器大小。如果 ``filter_size`` 是列表或元组，则它必须包含两个整数（filter_size_H，filter_size_W）。否则，filter_size_H = filter_size_W = filter_size。
    - **pool_size** （int | list | tuple） - Pool2d池化层大小。如果pool_size是列表或元组，则它必须包含两个整数（pool_size_H，pool_size_W）。否则，pool_size_H = pool_size_W = pool_size。
    - **pool_stride** （int | list | tuple） - Pool2d池化层步长。如果pool_stride是列表或元组，则它必须包含两个整数（pooling_stride_H，pooling_stride_W）。否则，pooling_stride_H = pooling_stride_W = pool_stride。
    - **pool_padding** （int | list | tuple） - Pool2d池化层的padding。如果pool_padding是列表或元组，则它必须包含两个整数（pool_padding_H，pool_padding_W）。否则，pool_padding_H = pool_padding_W = pool_padding。默认值为0。
    - **pool_type** （str） - 池化类型可以是 ``max-pooling`` 的 ``max`` 和平均池的 ``avg`` 。默认 ``max`` 。
    - **global_pooling** （bool）- 是否使用全局池。如果global_pooling = true，则忽略pool_size和pool_padding。默认为False
    - **conv_stride** （int | list | tuple） - conv2d Layer的步长。如果stride是列表或元组，则它必须包含两个整数，（conv_stride_H，conv_stride_W）。否则，conv_stride_H = conv_stride_W = conv_stride。默认值：conv_stride = 1。
    - **conv_padding** （int | list | tuple） - conv2d Layer的padding大小。如果padding是列表或元组，则它必须包含两个整数（conv_padding_H，conv_padding_W）。否则，conv_padding_H = conv_padding_W = conv_padding。默认值：conv_padding = 0。
    - **conv_dilation** （int | list | tuple） - conv2d Layer的dilation大小。如果dilation是列表或元组，则它必须包含两个整数（conv_dilation_H，conv_dilation_W）。否则，conv_dilation_H = conv_dilation_W = conv_dilation。默认值：conv_dilation = 1。
    - **conv_groups** （int） - conv2d Layer的组数。根据Alex Krizhevsky的Deep CNN论文中的分组卷积：当group = 2时，前半部分滤波器仅连接到输入通道的前半部分，而后半部分滤波器仅连接到后半部分输入通道。默认值：groups = 1。
    - **param_attr** （ParamAttr | None） - 可学习参数的参数属性或conv2d权重。如果将其设置为None或ParamAttr的一个属性，则conv2d将创建ParamAttr作为param_attr。如果未设置param_attr的初始化，则使用 :math:`Normal（0.0，std）` 初始化参数，并且 ``std`` 为 :math:`(\frac{2.0 }{filter\_elem\_num})^{0.5}` 。默认值:None
    - **bias_attr** （ParamAttr | bool | None） - conv2d的bias参数属性。如果设置为False，则不会向输出单元添加bias。如果将其设置为None或ParamAttr的一个属性，则conv2d将创建ParamAttr作为bias_attr。如果未设置bias_attr的初始化程序，则将偏差初始化为零。默认值：None
    - **act** （str） - conv2d的激活类型，如果设置为None，则不附加激活。默认值：无。
    - **use_cudnn** （bool） - 是否使用cudnn内核，仅在安装cudnn库时才有效。默认值：True。

返回： Convolution2d和Pool2d之后输入的结果。

返回类型：  变量（Variable）

**示例代码**

.. code-block:: python

    import paddle.fluid as fluid
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    conv_pool = fluid.nets.simple_img_conv_pool(input=img,
                                            filter_size=5,
                                            num_filters=20,
                                            pool_size=2,
                                            pool_stride=2,
                                            act="relu")











