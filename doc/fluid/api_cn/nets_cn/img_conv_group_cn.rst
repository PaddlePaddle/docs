.. _cn_api_fluid_nets_img_conv_group:

img_conv_group
-------------------------------

.. py:function:: paddle.fluid.nets.img_conv_group(input, conv_num_filter, pool_size, conv_padding=1, conv_filter_size=3, conv_act=None, param_attr=None, conv_with_batchnorm=False, conv_batchnorm_drop_rate=0.0, pool_stride=1, pool_type='max', use_cudnn=True)

Image Convolution Group由Convolution2d，BatchNorm，DropOut和Pool2d组成。根据输入参数，img_conv_group将使用Convolution2d，BatchNorm，DropOut对Input进行连续计算，并将最后一个结果传递给Pool2d。

参数：
       - **input** （Variable） - 具有[N，C，H，W]格式的输入图像。
       - **conv_num_filter** （list | tuple） - 表示该组的过滤器数。
       - **pool_size** （int | list | tuple） -  ``Pool2d Layer`` 池的大小。如果pool_size是列表或元组，则它必须包含两个整数（pool_size_H，pool_size_W）。否则，pool_size_H = pool_size_W = pool_size。
       - **conv_padding** （int | list | tuple） - Conv2d Layer的 ``padding`` 大小。如果 ``padding`` 是列表或元组，则其长度必须等于 ``conv_num_filter`` 的长度。否则，所有Conv2d图层的 ``conv_padding`` 都是相同的。默认1。
       - **conv_filter_size** （int | list | tuple） - 过滤器大小。如果filter_size是列表或元组，则其长度必须等于 ``conv_num_filter`` 的长度。否则，所有Conv2d图层的 ``conv_filter_size`` 都是相同的。默认3。
       - **conv_act** （str） -  ``Conv2d Layer`` 的激活类型， ``BatchNorm`` 后面没有。默认值：无。
       - **param_attr** （ParamAttr） - Conv2d层的参数。默认值：无
       - **conv_with_batchnorm** （bool | list） - 表示在 ``Conv2d Layer`` 之后是否使用 ``BatchNorm`` 。如果 ``conv_with_batchnorm`` 是一个列表，则其长度必须等于 ``conv_num_filter`` 的长度。否则， ``conv_with_batchnorm`` 指示是否所有Conv2d层都遵循 ``BatchNorm``。默认为False。
       - **conv_batchnorm_drop_rate** （float | list） - 表示 ``BatchNorm`` 之后的 ``Dropout Layer`` 的 ``rop_rate`` 。如果 ``conv_batchnorm_drop_rate`` 是一个列表，则其长度必须等于 ``conv_num_filter`` 的长度。否则，所有 ``Dropout Layers`` 的 ``drop_rate`` 都是   ``conv_batchnorm_drop_rate`` 。默认值为0.0。
       - **pool_stride** （int | list | tuple） -  ``Pool2d`` 层的汇集步幅。如果 ``pool_stride`` 是列表或元组，则它必须包含两个整数（pooling_stride_H，pooling_stride_W）。否则，pooling_stride_H = pooling_stride_W = pool_stride。默认1。
       - **pool_type** （str） - 池化类型可以是最大池化的 ``max`` 和平均池化的 ``avg`` 。默认max。
       - **use_cudnn** （bool） - 是否使用cudnn内核，仅在安装cudnn库时才有效。默认值：True
       
返回：  使用Convolution2d进行串行计算后的最终结果，BatchNorm，DropOut和Pool2d。

返回类型：  变量（Variable）。

**代码示例**

.. code-block:: python

          import paddle.fluid as fluid
          img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
          conv_pool = fluid.nets.img_conv_group(input=img,
                                                conv_padding=1,
                                                conv_num_filter=[3, 3],
                                                conv_filter_size=3,
                                                conv_act="relu",
                                                pool_size=2,
                                                pool_stride=2)







