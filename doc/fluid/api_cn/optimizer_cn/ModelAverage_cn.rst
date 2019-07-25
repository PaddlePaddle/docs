.. _cn_api_fluid_optimizer_ModelAverage:

ModelAverage
-------------------------------

.. py:class:: paddle.fluid.optimizer.ModelAverage(average_window_rate, min_average_window=10000, max_average_window=10000, regularization=None, name=None)

在滑动窗口中累积参数的平均值。平均结果将保存在临时变量中，通过调用 ``apply()`` 方法可应用于当前模型的参数变量。使用 ``restore()`` 方法恢复当前模型的参数值。

平均窗口的大小由 ``average_window_rate`` ， ``min_average_window`` ， ``max_average_window`` 以及当前更新次数决定。

 
参数:
  - **average_window_rate** – 窗口平均速率
  - **min_average_window** – 平均窗口大小的最小值
  - **max_average_window** – 平均窗口大小的最大值
  - **regularization** – 正则化器，例如 ``fluid.regularizer.L2DecayRegularizer`` 
  - **name** – 可选的名称前缀

**代码示例**

.. code-block:: python
        
    import paddle.fluid as fluid
    import numpy
     
    # 首先创建执行引擎
    place = fluid.CPUPlace()  # fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
     
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(train_program, startup_program):
        # 构建net
        data = fluid.layers.data(name='X', shape=[1], dtype='float32')
        hidden = fluid.layers.fc(input=data, size=10)
        loss = fluid.layers.mean(hidden)
        optimizer = fluid.optimizer.Momentum(learning_rate=0.2, momentum=0.1)
        optimizer.minimize(loss)

        # 构建ModelAverage优化器
        model_average = fluid.optimizer.ModelAverage(0.15,
                                          min_average_window=10000,
                                          max_average_window=20000)
        exe.run(startup_program)
        x = numpy.random.random(size=(10, 1)).astype('float32')
        outs = exe.run(program=train_program,
                       feed={'X': x},
                       fetch_list=[loss.name])
       # 应用ModelAverage
        with model_average.apply(exe):
             x = numpy.random.random(size=(10, 1)).astype('float32')
             exe.run(program=train_program,
                    feed={'X': x},
                    fetch_list=[loss.name])


.. py:method:: apply(executor, need_restore=True)

将平均值应用于当前模型的参数。

参数：
    - **executor** (fluid.Executor) – 当前的执行引擎。
    - **need_restore** (bool) – 如果您最后需要实现恢复，将其设为True。默认值True。


.. py:method:: restore(executor)

恢复当前模型的参数值

参数：
    - **executor** (fluid.Executor) – 当前的执行引擎。






