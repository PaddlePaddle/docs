.. _cn_user_guide_Operator:

=========
Operator
=========

在飞桨（PaddlePaddle，以下简称Paddle）中，所有对数据的操作都由Operator表示

为了便于用户使用，在Python端，Paddle中的Operator被封装入 :code:`paddle.fluid.layers` ， :code:`paddle.fluid.nets` 等模块。

因为一些常见的对Tensor的操作可能是由更多基础操作构成，为了提高使用的便利性，框架内部对基础 Operator 进行了一些封装，包括创建 Operator 依赖可学习参数，可学习参数的初始化细节等，减少用户重复开发的成本。

例如用户可以利用 :code:`paddle.fluid.layers.elementwise_add()` 实现两个输入Tensor的加法运算：

.. code-block:: python

    import paddle.fluid as fluid
    import numpy

    a = fluid.data(name="a",shape=[1],dtype='float32')
    b = fluid.data(name="b",shape=[1],dtype='float32')

    result = fluid.layers.elementwise_add(a,b)

    # 定义执行器，并且制定执行的设备为CPU
    cpu = fluid.core.CPUPlace()
    exe = fluid.Executor(cpu)
    exe.run(fluid.default_startup_program())

    x = numpy.array([5]).astype("float32")
    y = numpy.array([7]).astype("float32")

    outs = exe.run(
            feed={'a':x,'b':y},
            fetch_list=[result])
    # 打印输出结果，[array([12.], dtype=float32)]
    print( outs )


如果想获取网络执行过程中的a，b的具体值，可以将希望查看的变量添加在fetch_list中。

.. code-block:: python

    #执行计算
    outs = exe.run(
        feed={'a':x,'b':y},
        fetch_list=[a,b,result])
    #查看输出结果
    print( outs )


输出结果：

.. code-block:: python

    [array([5.], dtype=float32), array([7.], dtype=float32), array([12.], dtype=float32)]

