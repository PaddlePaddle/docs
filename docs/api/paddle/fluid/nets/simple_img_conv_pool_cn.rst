.. _cn_api_fluid_nets_simple_img_conv_pool:

simple_img_conv_pool
-------------------------------


.. py:function:: paddle.fluid.nets.simple_img_conv_pool(input, num_filters, filter_size, pool_size, pool_stride, pool_padding=0, pool_type='max', global_pooling=False, conv_stride=1, conv_padding=0, conv_dilation=1, conv_groups=1, param_attr=None, bias_attr=None, act=None, use_cudnn=True)




 ``simple_img_conv_pool`` 由一个conv2d( :ref:`cn_api_fluid_layers_conv2d` )和一个pool2d( :ref:`cn_api_fluid_layers_pool2d` ) OP组成。

参数
::::::::::::

    - **input** (Variable) - 输入图像，4-D Tensor，格式为[N，C，H，W]。数据类型是float32或者float64
    - **num_filters** (int) - 卷积核的数目，整数。
    - **filter_size** (int | list | tuple) - conv2d卷积核大小，整数或者整型列表或者整型元组。如果 ``filter_size`` 是列表或元组，则它必须包含两个整数(filter_size_H，filter_size_W)。如果是整数，则filter_size_H = filter_size_W = filter_size。
    - **pool_size** (int | list | tuple) - pool2d池化层大小，整数或者整型列表或者整型元组。如果pool_size是列表或元组，则它必须包含两个整数(pool_size_H，pool_size_W)。如果是整数，则pool_size_H = pool_size_W = pool_size。
    - **pool_stride** (int | list | tuple) - pool2d池化层步长，整数或者整型列表或者整型元组。如果pool_stride是列表或元组，则它必须包含两个整数(pooling_stride_H，pooling_stride_W)。如果是整数，pooling_stride_H = pooling_stride_W = pool_stride。
    - **pool_padding** (int | list | tuple，可选) - pool2d池化层的padding，整数或者整型列表或者整型元组。如果pool_padding是列表或元组，则它必须包含两个整数(pool_padding_H，pool_padding_W)。如果是整数，pool_padding_H = pool_padding_W = pool_padding。默认值为0。
    - **pool_type** (str，可选) - 池化类型，字符串，可以是 ``max`` 或者 ``avg``，分别对应最大池化和平均池化。默认 ``max`` 。
    - **global_pooling** (bool，可选)- 是否使用全局池化。如果global_pooling = true，则忽略pool_size和pool_padding。默认为False
    - **conv_stride** (int | list | tuple，可选) - conv2d Layer的卷积步长，整数或者整型列表或者整型元组。如果conv_stride是列表或元组，则它必须包含两个整数，(conv_stride_H，conv_stride_W)。如果是整数，conv_stride_H = conv_stride_W = conv_stride。默认值：conv_stride = 1。
    - **conv_padding** (int | list | tuple，可选) - conv2d Layer的padding大小，整数或者整型列表或者整型元组。如果conv_padding是列表或元组，则它必须包含两个整数(conv_padding_H，conv_padding_W)。如果是整数，conv_padding_H = conv_padding_W = conv_padding。默认值：conv_padding = 0。
    - **conv_dilation** (int | list | tuple，可选) - conv2d Layer的dilation大小，整数或者整型列表或者整型元。如果conv_dilation是列表或元组，则它必须包含两个整数(conv_dilation_H，conv_dilation_W)。如果是整数，conv_dilation_H = conv_dilation_W = conv_dilation。默认值：conv_dilation = 1。
    - **conv_groups** (int，可选) - conv2d Layer的组数，整数。根据Alex Krizhevsky的Deep CNN论文中的分组卷积：当group = 2时，前半部分滤波器仅连接到输入通道的前半部分，而后半部分滤波器仅连接到后半部分输入通道。默认值：conv_groups = 1。
    - **param_attr** (ParamAttr，可选) - conv2d的weights参数属性。如果将其设置为None或ParamAttr的一个属性，则conv2d将创建ParamAttr作为param_attr。如果未设置param_attr的初始化，则使用 :math:`Normal（0.0，std）` 初始化参数，并且 ``std`` 为 :math:`(\frac{2.0 }{filter\_elem\_num})^{0.5}`。默认值：None
    - **bias_attr** (ParamAttr | bool | None，可选) - conv2d的bias参数属性。如果设置为False，则不会向输出单元添加bias。如果将其设置为None或ParamAttr的一个属性，则conv2d将创建ParamAttr作为bias_attr。如果设置bias_attr为None，则将其初始化为零。默认值：None
    - **act** (str，可选) - conv2d的激活类型，字符串，可以是'relu', 'softmax', 'sigmoid'等激活函数的类型。如果设置为None，则不附加激活。默认值：None。
    - **use_cudnn** (bool，可选) - 是否使用cudnn内核，仅在安装cudnn库时才有效。默认值：True。
    - **name** (str|None，可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name`，默认值为None

返回
::::::::::::
 输入input经过conv2d和pool2d之后输入的结果，数据类型与input相同

返回类型
::::::::::::
  Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.nets.simple_img_conv_pool