.. _cn_api_fluid_dygraph_Pool2D:

Pool2D
-------------------------------

.. py:class:: paddle.fluid.dygraph.Pool2D(name_scope, pool_size=-1, pool_type='max', pool_stride=1, pool_padding=0, global_pooling=False, use_cudnn=True, ceil_mode=False, exclusive=True, dtype=VarType.FP32)

pooling2d操作符根据 ``input`` ， 池化类型 ``pooling_type`` ， 池化核大小 ``ksize`` , 步长 ``strides`` ，填充 ``paddings`` 这些参数得到输出。

输入X和输出Out是NCHW格式，N为batch尺寸，C是通道数，H是特征高度，W是特征宽度。

参数（ksize,strides,paddings）含有两个元素。这两个元素分别代表高度和宽度。输入X的大小和输出Out的大小可能不一致。


参数：
    - **name_scope** (str) - 该类的名称
    - **pool_size** (int|list|tuple)  - 池化核的大小。如果它是一个元组或列表，它必须包含两个整数值， (pool_size_Height, pool_size_Width)。其他情况下，若为一个整数，则它的平方值将作为池化核大小，比如若pool_size=2, 则池化核大小为2x2，默认值为-1。
    - **pool_type** (string) - 池化类型，可以是“max”对应max-pooling，“avg”对应average-pooling，默认值为max。
    - **pool_stride** (int|list|tuple)  - 池化层的步长。如果它是一个元组或列表，它将包含两个整数，(pool_stride_Height, pool_stride_Width)。否则它是一个整数的平方值。默认值为1。
    - **pool_padding** (int|list|tuple) - 填充大小。如果它是一个元组，它必须包含两个整数值，(pool_padding_on_Height, pool_padding_on_Width)。否则它是一个整数的平方值。默认值为0。
    - **global_pooling** （bool）- 是否用全局池化。如果global_pooling = true， ``ksize`` 和 ``paddings`` 将被忽略。默认值为false
    - **use_cudnn** （bool）- 只在cudnn核中用，需要安装cudnn，默认值为True。
    - **ceil_mode** （bool）- 是否用ceil函数计算输出高度和宽度。默认False。如果设为False，则使用floor函数。默认值为false。
    - **exclusive** (bool) - 是否在平均池化模式忽略填充值。默认为True。

返回：池化结果

返回类型：变量（Variable）

抛出异常：
    - ``ValueError`` - 如果 ``pool_type`` 既不是“max”也不是“avg”
    - ``ValueError`` - 如果 ``global_pooling`` 为False并且‘pool_size’为-1
    - ``ValueError`` - 如果 ``use_cudnn`` 不是bool值

**代码示例**

.. code-block:: python

          import paddle.fluid as fluid
          import numpy

          with fluid.dygraph.guard():
             data = numpy.random.random((3, 32, 32)).astype('float32')

             pool2d = fluid.dygraph.Pool2D("pool2d",pool_size=2,
                            pool_type='max',
                            pool_stride=1,
                            global_pooling=False)
             pool2d_res = pool2d(data)





