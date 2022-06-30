.. _cn_api_fluid_nets_img_conv_group:

img_conv_group
-------------------------------


.. py:function:: paddle.fluid.nets.img_conv_group(input, conv_num_filter, pool_size, conv_padding=1, conv_filter_size=3, conv_act=None, param_attr=None, conv_with_batchnorm=False, conv_batchnorm_drop_rate=0.0, pool_stride=1, pool_type='max', use_cudnn=True)




Image Convolution Group由Convolution2d，BatchNorm，DropOut和Pool2d组成。根据输入参数，img_conv_group将使用Convolution2d，BatchNorm，DropOut对Input进行连续计算，得到最后结果。

参数
::::::::::::

       - **input** （Variable） - 输入，格式为[N，C，H，W]的4-D Tensor。数据类型：float32和float64。
       - **conv_num_filter** （list | tuple） - 卷积中使用的滤波器数。
       - **pool_size** （int | list | tuple） - 池化层中池化核的大小。如果pool_size是列表或元组，则它必须包含两个整数（pool_size_height，pool_size_width）。否则，pool_size_height = pool_size_width = pool_size。
       - **conv_padding** （int | list | tuple） - 卷积层中的填充 ``padding`` 的大小。如果 ``padding`` 是列表或元组，则其长度必须等于 ``conv_num_filter`` 的长度。否则，所有卷积的 ``conv_padding`` 都是相同的。默认：1。
       - **conv_filter_size** （int | list | tuple） - 卷积层中滤波器大小。如果filter_size是列表或元组，则其长度必须等于 ``conv_num_filter`` 的长度。否则，所有卷积的 ``conv_filter_size`` 都是相同的。默认：3。
       - **conv_act** （str） -  卷积层之后接的的激活层类型，``BatchNorm`` 后面没有。默认：None。
       - **param_attr** (ParamAttr|None)：指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。conv2d算子默认的权重初始化是Xavier。
       - **conv_with_batchnorm** （bool | list） - 表示在卷积层之后是否使用 ``BatchNorm``。如果 ``conv_with_batchnorm`` 是一个列表，则其长度必须等于 ``conv_num_filter`` 的长度。否则，``conv_with_batchnorm`` 指示是否所有卷积层后都使用 ``BatchNorm``。默认：False。
       - **conv_batchnorm_drop_rate** （float | list） - 表示 ``BatchNorm`` 之后的 ``Dropout Layer`` 的 ``drop_rate``。如果 ``conv_batchnorm_drop_rate`` 是一个列表，则其长度必须等于 ``conv_num_filter`` 的长度。否则，所有 ``Dropout Layers`` 的 ``drop_rate`` 都是   ``conv_batchnorm_drop_rate``。默认：0.0。
       - **pool_stride** （int | list | tuple） -  池化层的池化步长。如果 ``pool_stride`` 是列表或元组，则它必须包含两个整数（pooling_stride_height，pooling_stride_width）。否则，pooling_stride_height = pooling_stride_width = pool_stride。默认：1。
       - **pool_type** （str） - 池化类型可以是最大池化的 ``max`` 和平均池化的 ``avg``。默认：max。
       - **use_cudnn** （bool） - 是否使用cudnn内核，仅在安装cudnn库时才有效。默认值：True
       
返回
::::::::::::
 Tensor。使用Convolution2d，BatchNorm，DropOut和Pool2d进行串行计算后的最终结果。

返回类型
::::::::::::
 Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.nets.img_conv_group