.. _cn_api_fluid_optimizer_MomentumOptimizer:

MomentumOptimizer
-------------------------------

.. py:class::  paddle.fluid.optimizer.MomentumOptimizer(learning_rate, momentum, use_nesterov=False, regularization=None, name=None)

该接口实现含有速度状态的Simple Momentum 优化器

该优化器含有牛顿动量标志，公式更新如下：

.. math::
    & velocity = mu * velocity + gradient\\
    & if (use\_nesterov):\\
    &\quad   param = param - (gradient + mu * velocity) * learning\_rate\\
    & else:\\&\quad   param = param - learning\_rate * velocity

参数：
    - **learning_rate** (float|Variable) - 学习率，用于参数更新。作为数据参数，可以是浮点型值或含有一个浮点型值的变量。
    - **momentum** (float) - 动量因子。
    - **use_nesterov** (bool，可选) - 赋能牛顿动量，默认值False。
    - **regularization** - 正则化函数，，例如 :code:`fluid.regularizer.L2DecayRegularizer`，默认值None。
    - **name** (str, 可选) - 可选的名称前缀，一般无需设置，默认值为None。

**代码示例**：

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np

    place = fluid.CPUPlace()
    main = fluid.Program()
    with fluid.program_guard(main):
        x = fluid.layers.data(name='x', shape=[13], dtype='float32')
        y = fluid.layers.data(name='y', shape=[1], dtype='float32')
        y_predict = fluid.layers.fc(input=x, size=1, act=None)
        cost = fluid.layers.square_error_cost(input=y_predict, label=y)
        avg_cost = fluid.layers.mean(cost)

        moment_optimizer = fluid.optimizer.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
        moment_optimizer.minimize(avg_cost)

        fetch_list = [avg_cost]
        train_reader = paddle.batch(
            paddle.dataset.uci_housing.train(), batch_size=1)
        feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        for data in train_reader():
            exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)



.. py:method:: minimize(loss, startup_program=None, parameter_list=None, no_grad_set=None, grad_clip=None)


通过更新parameter_list来添加操作，进而使损失最小化。

该算子相当于backward()和apply_gradients()功能的合体。

参数：
    - **loss** (Variable) – 用于优化过程的损失值变量。
    - **startup_program** (Program) – 用于初始化在parameter_list中参数的startup_program。
    - **parameter_list** (list) – 待更新的Variables组成的列表。
    - **no_grad_set** (set|None) – 应该被无视的Variables集合。
    - **grad_clip** (GradClipBase|None) – 梯度裁剪的策略。

返回： 附加的算子列表和由(param, grad) 变量对组成的元组，用于优化。

返回类型：tuple(optimize_ops, params_grads)




