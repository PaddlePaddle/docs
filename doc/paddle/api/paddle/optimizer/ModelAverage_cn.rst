.. _cn_api_fluid_optimizer_ModelAverage:

ModelAverage
-------------------------------


.. py:class:: paddle.fluid.optimizer.ModelAverage(average_window_rate, min_average_window=10000, max_average_window=10000, regularization=None, name=None)

:api_attr: 声明式编程模式（静态图)



ModelAverage优化器，在训练过程中累积特定连续的历史Parameters，累积的历史范围可以用传入的average_window参数来控制，在预测时使用平均后的Parameters，通常可以提高预测的精度。

在滑动窗口中累积Parameters的平均值，将结果将保存在临时变量中，通过调用 ``apply()`` 方法可应用于当前模型的Parameters，使用 ``restore()`` 方法恢复当前模型Parameters的值。

计算平均值的窗口大小由 ``average_window_rate`` ， ``min_average_window`` ， ``max_average_window`` 以及当前Parameters更新次数(num_updates)共同决定。

累积次数（num_accumulates）大于特定窗口阈值(average_window)时，将累积的Parameters临时变量置为0.0，这几个参数的作用通过以下示例代码说明：

.. code-block:: python

    if num_accumulates >= min_average_window and num_accumulates >= min(max_average_window, num_updates * average_window_rate):
        num_accumulates = 0

上述条件判断语句中，num_accumulates表示当前累积的次数，可以抽象理解为累积窗口的长度，窗口长度至少要达到min_average_window参数设定的长度，并且不能超过max_average_window参数或者num_updates * average_window_rate规定的长度，其中num_updates表示当前Parameters更新的次数，average_window_rate是一个计算窗口长度的系数。
 
参数：
  - **average_window_rate** (float) – 相对于Parameters更新次数的窗口长度计算比率
  - **min_average_window** (int, 可选) – 平均值计算窗口长度的最小值，默认值为10000
  - **max_average_window** (int, 可选) – 平均值计算窗口长度的最大值，推荐设置为一轮训练中mini-batchs的数目，默认值为10000
  - **regularization** (WeightDecayRegularizer，可选) - 正则化方法。支持两种正则化策略: :ref:`cn_api_fluid_regularizer_L1Decay` 、 
    :ref:`cn_api_fluid_regularizer_L2Decay` 。如果一个参数已经在 :ref:`cn_api_fluid_ParamAttr` 中设置了正则化，这里的正则化设置将被忽略；
    如果没有在 :ref:`cn_api_fluid_ParamAttr` 中设置正则化，这里的设置才会生效。默认值为None，表示没有正则化。
  - **name** (str, 可选)– 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None

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
                                          max_average_window=12500)
        exe.run(startup_program)
        for i in range(12500):
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

将累积Parameters的平均值应用于当前网络的Parameters。

参数：
    - **executor** (fluid.Executor) – 当前网络的执行器
    - **need_restore** (bool) – 恢复标志变量，设为True时，执行完成后会将网络的Parameters恢复为网络默认的值，设为False将不会恢复，默认值True

返回：无

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
                                          max_average_window=12500)
        exe.run(startup_program)
        for i in range(12500):
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

.. py:method:: restore(executor)

恢复当前网络的Parameters值

参数：
    - **executor** (fluid.Executor) – 当前网络的执行器

返回：无

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
                                          max_average_window=12500)
        exe.run(startup_program)
        for i in range(12500):
            x = numpy.random.random(size=(10, 1)).astype('float32')
            outs = exe.run(program=train_program,
                        feed={'X': x},
                        fetch_list=[loss.name])

       # 应用ModelAverage
        with model_average.apply(exe, False):
             x = numpy.random.random(size=(10, 1)).astype('float32')
             exe.run(program=train_program,
                    feed={'X': x},
                    fetch_list=[loss.name])
        # 恢复网络Parameters
        model_average.restore(exe)
