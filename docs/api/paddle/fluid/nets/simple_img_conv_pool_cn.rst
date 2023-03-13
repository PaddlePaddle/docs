.. _cn_api_fluid_nets_simple_img_conv_pool:

simple_img_conv_pool
-------------------------------


.. py:function:: paddle.fluid.nets.simple_img_conv_pool(input, num_filters, filter_size, pool_size, pool_stride, pool_padding=0, pool_type='max', global_pooling=False, conv_stride=1, conv_padding=0, conv_dilation=1, conv_groups=1, param_attr=None, bias_attr=None, act=None, use_cudnn=True)




 ``simple_img_conv_pool`` 由一个 conv2d( :ref:`cn_api_fluid_layers_conv2d` )和一个 pool2d( :ref:`cn_api_fluid_layers_pool2d` ) OP 组成。

参数
::::::::::::

    - **input** (Variable) - 输入图像，4-D Tensor，格式为[N，C，H，W]。数据类型是 float32 或者 float64
    - **num_filters** (int) - 卷积核的数目，整数。
    - **filter_size** (int | list | tuple) - conv2d 卷积核大小，整数或者整型列表或者整型元组。如果 ``filter_size`` 是列表或元组，则它必须包含两个整数(filter_size_H，filter_size_W)。如果是整数，则 filter_size_H = filter_size_W = filter_size。
    - **pool_size** (int | list | tuple) - pool2d 池化层大小，整数或者整型列表或者整型元组。如果 pool_size 是列表或元组，则它必须包含两个整数(pool_size_H，pool_size_W)。如果是整数，则 pool_size_H = pool_size_W = pool_size。
    - **pool_stride** (int | list | tuple) - pool2d 池化层步长，整数或者整型列表或者整型元组。如果 pool_stride 是列表或元组，则它必须包含两个整数(pooling_stride_H，pooling_stride_W)。如果是整数，pooling_stride_H = pooling_stride_W = pool_stride。
    - **pool_padding** (int | list | tuple，可选) - pool2d 池化层的 padding，整数或者整型列表或者整型元组。如果 pool_padding 是列表或元组，则它必须包含两个整数(pool_padding_H，pool_padding_W)。如果是整数，pool_padding_H = pool_padding_W = pool_padding。默认值为 0。
    - **pool_type** (str，可选) - 池化类型，字符串，可以是 ``max`` 或者 ``avg``，分别对应最大池化和平均池化。默认 ``max`` 。
    - **global_pooling** (bool，可选)- 是否使用全局池化。如果 global_pooling = true，则忽略 pool_size 和 pool_padding。默认为 False
    - **conv_stride** (int | list | tuple，可选) - conv2d Layer 的卷积步长，整数或者整型列表或者整型元组。如果 conv_stride 是列表或元组，则它必须包含两个整数，(conv_stride_H，conv_stride_W)。如果是整数，conv_stride_H = conv_stride_W = conv_stride。默认值：conv_stride = 1。
    - **conv_padding** (int | list | tuple，可选) - conv2d Layer 的 padding 大小，整数或者整型列表或者整型元组。如果 conv_padding 是列表或元组，则它必须包含两个整数(conv_padding_H，conv_padding_W)。如果是整数，conv_padding_H = conv_padding_W = conv_padding。默认值：conv_padding = 0。
    - **conv_dilation** (int | list | tuple，可选) - conv2d Layer 的 dilation 大小，整数或者整型列表或者整型元。如果 conv_dilation 是列表或元组，则它必须包含两个整数(conv_dilation_H，conv_dilation_W)。如果是整数，conv_dilation_H = conv_dilation_W = conv_dilation。默认值：conv_dilation = 1。
    - **conv_groups** (int，可选) - conv2d Layer 的组数，整数。根据 Alex Krizhevsky 的 Deep CNN 论文中的分组卷积：当 group = 2 时，前半部分滤波器仅连接到输入通道的前半部分，而后半部分滤波器仅连接到后半部分输入通道。默认值：conv_groups = 1。
    - **param_attr** (ParamAttr，可选) - conv2d 的 weights 参数属性。如果将其设置为 None 或 ParamAttr 的一个属性，则 conv2d 将创建 ParamAttr 作为 param_attr。如果未设置 param_attr 的初始化，则使用 :math:`Normal（0.0，std）` 初始化参数，并且 ``std`` 为 :math:`(\frac{2.0 }{filter\_elem\_num})^{0.5}`。默认值：None
    - **bias_attr** (ParamAttr | bool | None，可选) - conv2d 的 bias 参数属性。如果设置为 False，则不会向输出单元添加 bias。如果将其设置为 None 或 ParamAttr 的一个属性，则 conv2d 将创建 ParamAttr 作为 bias_attr。如果设置 bias_attr 为 None，则将其初始化为零。默认值：None
    - **act** (str，可选) - conv2d 的激活类型，字符串，可以是'relu', 'softmax', 'sigmoid'等激活函数的类型。如果设置为 None，则不附加激活。默认值：None。
    - **use_cudnn** (bool，可选) - 是否使用 cudnn 内核，仅在安装 cudnn 库时才有效。默认值：True。
    - **name** (str|None，可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name`，默认值为 None

返回
::::::::::::
 输入 input 经过 conv2d 和 pool2d 之后输入的结果，数据类型与 input 相同

返回类型
::::::::::::
  Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.nets.simple_img_conv_pool
