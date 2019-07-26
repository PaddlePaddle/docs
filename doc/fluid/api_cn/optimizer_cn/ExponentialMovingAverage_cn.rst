.. _cn_api_fluid_optimizer_ExponentialMovingAverage:

ExponentialMovingAverage
-------------------------------

.. py:class:: paddle.fluid.optimizer.ExponentialMovingAverage(decay=0.999, thres_steps=None, name=None)

用指数衰减计算参数的移动平均值。
给出参数 :math:`\theta` ，它的指数移动平均值(exponential moving average, EMA)
为

.. math::
    \begin{align}\begin{aligned}\text{EMA}_0 & = 0\\\text{EMA}_t & = \text{decay} * \text{EMA}_{t-1} + (1 - \text{decay}) * \theta_t\end{aligned}\end{align}


用 ``update()`` 方法计算出的平均结果将保存在由对象创建和维护的临时变量中，并且可以通过调用 ``apply()`` 方法把结果应用于当前模型的参数。另外，``restore()`` 方法用于恢复参数。

**偏差教正。**  所有的EMAs均初始化为 :math:`0` ，因此它们将为零偏差，可以通过除以因子 :math:`(1 - \text{decay}^t)` 来校正，即在调用 ``apply()`` 方法时应用于参数的真实EMAs将为：

.. math::
    \widehat{\text{EMA}}_t = \frac{\text{EMA}_t}{1 - \text{decay}^t}

**衰减率调度。**  一个非常接近于1的很大的衰减率将会导致平均值移动得很慢。更优的策略是，一开始就设置一个相对较小的衰减率。参数thres_steps允许用户传递一个变量以设置衰减率，在这种情况下，
真实的衰减率变为 ：

.. math:: 
    \min(\text{decay}, \frac{1 + \text{thres_steps}}{10 + \text{thres_steps}})

通常thres_steps可以是全局训练steps。
     

参数：
    - **decay** (float) – 指数衰减率，通常接近1，如0.999，0.9999，……
    - **thres_steps** (Variable|None) – 如果不为None，指定衰减率。
    - **name** (str|None) – 名字前缀（可选项）。

**代码示例**

.. code-block:: python

    import numpy
    import paddle
    import paddle.fluid as fluid

    data = fluid.layers.data(name='x', shape=[5], dtype='float32')
    hidden = fluid.layers.fc(input=data, size=10)
    cost = fluid.layers.mean(hidden)

    test_program = fluid.default_main_program().clone(for_test=True)

    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    optimizer.minimize(cost)

    global_steps = fluid.layers.learning_rate_scheduler._decay_step_counter()
    ema = fluid.optimizer.ExponentialMovingAverage(0.999, thres_steps=global_steps)
    ema.update()

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    for pass_id in range(3):
        for batch_id in range(6):
            data = numpy.random.random(size=(10, 5)).astype('float32')
            exe.run(program=fluid.default_main_program(),
                feed={'x': data},
                fetch_list=[cost.name])

        # usage 1
        with ema.apply(exe):
            data = numpy.random.random(size=(10, 5)).astype('float32')
            exe.run(program=test_program,
                    feed={'x': data},
                    fetch_list=[hidden.name])


         # usage 2
        with ema.apply(exe, need_restore=False):
            data = numpy.random.random(size=(10, 5)).astype('float32')
            exe.run(program=test_program,
                    feed={'x': data},
                    fetch_list=[hidden.name])
        ema.restore(exe)


.. py:method:: update()

更新指数滑动平均。仅在训练程序中调用此方法。

.. py:method:: apply(executor, need_restore=True)

参数：
    - **executor** (Executor) – 执行应用的执行引擎。
    - **need_restore** (bool) –是否在应用后恢复参数。

.. py:method:: restore(executor)

参数：
    - **executor** (Executor) – 执行存储的执行引擎。




