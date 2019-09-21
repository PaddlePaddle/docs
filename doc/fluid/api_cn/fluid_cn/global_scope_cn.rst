.. _cn_api_fluid_global_scope:

global_scope
-------------------------------

.. py:function:: paddle.fluid.global_scope()


该接口用于返回Paddle内部用来做全局变量管理的 ``global scope`` 

``global scope`` 会存储运行过程中始终不会释放的全局variable: 
    -  `fluid.layers.data <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/layers_cn/data_cn.html#data>`_  创建的variable 
    - ``persistable`` 属性为True的variable
    - ...

参数：
    - 无

返回：用于管理全局变量的 ``global scope``

返回类型：`Scope <https://github.com/PaddlePaddle/Paddle/blob/cb65439da8b405c4a44c519276115cca1b7fef52/paddle/fluid/pybind/pybind.cc#L627>`_

**示例代码**

.. code-block:: python

        import paddle.fluid as fluid
        import paddle.fluid.compiler as compiler
        import numpy
        import os
        
        exe = fluid.Executor(fluid.CPUPlace())
        
        train_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name="x", shape=[2, 3], dtype='float32')
            y = fluid.layers.data(name="y", shape=[3], dtype='float32')
            z = fluid.layers.elementwise_add(x, y)
            z.persistable = True
        
            exe.run(startup_program)
        
            compiled_prog = compiler.CompiledProgram(train_program).with_data_parallel()
            res = exe.run(compiled_prog, feed=
            {"x": numpy.random.random(size=(2, 3)).astype('float32'),
             "y": numpy.random.random(size=(3)).astype('float32')},
                    fetch_list=[x, y, z]) 
        
            print(numpy.array_equal(numpy.array(fluid.global_scope().find_var(x.name).get_tensor()), res[0]))
            print(numpy.array_equal(numpy.array(fluid.global_scope().find_var(y.name).get_tensor()), res[1]))
            print(numpy.array_equal(numpy.array(fluid.global_scope().find_var(z.name).get_tensor()), res[2]))



