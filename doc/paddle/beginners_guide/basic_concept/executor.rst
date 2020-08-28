.. _cn_user_guide_Executor:

=======
Executor
=======


飞桨（PaddlePaddle，以下简称Paddle）的设计思想类似于高级编程语言C++和JAVA等。程序的执行过程被分为编译和执行两个阶段。

用户完成对 Program 的定义后，Executor 接受这段 Program 并转化为C++后端真正可执行的 FluidProgram，这一自动完成的过程叫做编译。

编译过后需要 Executor 来执行这段编译好的 FluidProgram。

例如上文实现的加法运算，当构建好 Program 后，需要创建 Executor，执行startup Program 和训练 Program：

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
