.. _cn_api_fluid_optimizer_FtrlOptimizer:

FtrlOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.FtrlOptimizer(learning_rate, l1=0.0, l2=0.0, lr_power=-0.5,regularization=None, name=None)
 
FTRL (Follow The Regularized Leader) Optimizer.

FTRL 原始论文: ( `https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf <https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf>`_)


.. math::
           &\qquad new\_accum=squared\_accum+grad^2\\\\
           &\qquad if(lr\_power==−0.5):\\
           &\qquad \qquad linear\_accum+=grad-\frac{\sqrt{new\_accum}-\sqrt{squared\_accum}}{learning\_rate*param}\\
           &\qquad else:\\
           &\qquad \qquad linear\_accum+=grad-\frac{new\_accum^{-lr\_power}-accum^{-lr\_power}}{learning\_rate*param}\\\\
           &\qquad x=l1*sign(linear\_accum)−linear\_accum\\\\
           &\qquad if(lr\_power==−0.5):\\
           &\qquad \qquad y=\frac{\sqrt{new\_accum}}{learning\_rate}+(2*l2)\\
           &\qquad \qquad pre\_shrink=\frac{x}{y}\\
           &\qquad \qquad param=(abs(linear\_accum)>l1).select(pre\_shrink,0.0)\\
           &\qquad else:\\
           &\qquad \qquad y=\frac{new\_accum^{-lr\_power}}{learning\_rate}+(2*l2)\\
           &\qquad \qquad pre\_shrink=\frac{x}{y}\\
           &\qquad \qquad param=(abs(linear\_accum)>l1).select(pre\_shrink,0.0)\\\\
           &\qquad squared\_accum+=grad^2


参数:
  - **learning_rate** (float|Variable)-全局学习率。
  - **l1** (float) - L1 regularization strength.
  - **l2** (float) - L2 regularization strength.
  - **lr_power** (float) - 学习率降低指数
  - **regularization** - 正则化器，例如 ``fluid.regularizer.L2DecayRegularizer`` 
  - **name** — 可选的名称前缀

抛出异常：
  - ``ValueError`` - 如果 ``learning_rate`` , ``rho`` ,  ``epsilon`` , ``momentum``  为 None.

**代码示例**

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
    
        ftrl_optimizer = fluid.optimizer.Ftrl(learning_rate=0.1)
        ftrl_optimizer.minimize(avg_cost)
    
        fetch_list = [avg_cost]
        train_reader = paddle.batch(
            paddle.dataset.uci_housing.train(), batch_size=1)
        feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        for data in train_reader():
            exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)


.. note::
     目前, FtrlOptimizer 不支持 sparse parameter optimization



.. py:method:: apply_gradients(params_grads)

为给定的params_grads对附加优化算子，为minimize过程的第二步

参数：
    - **params_grads** (list)- 用于优化的(param, grad)对组成的列表

返回：  附加在当前Program的算子组成的列表

返回类型：  list

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    loss = network()
    optimizer = fluid.optimizer.SGD(learning_rate=0.1)
    params_grads = optimizer.backward(loss)
    # you may append operations for params_grads here
    # ...
    optimizer.apply_gradients(params_grads)


.. py:method:: apply_optimize(loss, startup_program, params_grads)

为给定的params_grads对附加优化算子，为minimize过程的第二步。

参数：
    - **loss** (Variable) – 用于优化过程的损失值变量
    - **startup_program** (Program) – 用于初始化在parameter_list中参数的startup_program
    - **params_grads** (list)- 用于优化的(param, grad)对组成的列表

返回：  附加在当前Program的算子组成的列表

返回类型：  list

.. py:method:: backward(loss, startup_program=None, parameter_list=None, no_grad_set=None, callbacks=None)

自动做diff来向当前program附加反向算子，为minimize过程的第一步。

参数：
    - **loss** (Variable) – 用于优化过程的损失值变量
    - **startup_program** (Program) – 用于初始化在parameter_list中参数的startup_program
    - **parameter_list** (list) – 待更新的Variables组成的列表
    - **no_grad_set** (set|None) – 应该被无视的Variables集合
    - **callbacks** (list|None) – 当为某参数附加反向算子时所要运行的callables组成的列表

返回：  附加在当前Program的算子组成的列表

返回类型：  list

**代码示例**

详见apply_gradients的示例


.. py:method:: load(stat_dict)

在dygraph模式下，附带学习率衰减来加载优化器。

参数：
    - **stat_dict** – load_persistable方法加载的dict

**代码示例**

.. code-block:: python

    from __future__ import print_function
    import numpy as np
    import paddle
    import paddle.fluid as fluid
    from paddle.fluid.optimizer import SGDOptimizer
    from paddle.fluid.dygraph.nn import FC
    from paddle.fluid.dygraph.base import to_variable

    class MLP(fluid.Layer):
        def __init__(self, name_scope):
            super(MLP, self).__init__(name_scope)

            self._fc1 = FC(self.full_name(), 10)
            self._fc2 = FC(self.full_name(), 10)

        def forward(self, inputs):
            y = self._fc1(inputs)
            y = self._fc2(y)
            return y

    with fluid.dygraph.guard():
        mlp = MLP('mlp')
        optimizer2 = SGDOptimizer(
            learning_rate=fluid.layers.natural_exp_decay(
            learning_rate=0.1,
            decay_steps=10000,
            decay_rate=0.5,
            staircase=True))

        train_reader = paddle.batch(
                paddle.dataset.mnist.train(), batch_size=128, drop_last=True)

        for batch_id, data in enumerate(train_reader()):
            dy_x_data = np.array(
                    [x[0].reshape(1, 28, 28) for x in data]).astype('float32')

            y_data = np.array([x[1] for x in data]).astype('int64').reshape(
                    128, 1)

            img = to_variable(dy_x_data)
            label = to_variable(y_data)
            label._stop_gradient = True
            cost = mlp(img)
            avg_loss = fluid.layers.reduce_mean(cost)
            avg_loss.backward()
            optimizer.minimize(avg_loss)
            mlp.clear_gradients()
            fluid.dygraph.save_persistables(
                    mlp.state_dict(), [optimizer, optimizer2], "save_dir_2")
            if batch_id == 2:
                    break

    with fluid.dygraph.guard():
        mlp_load = MLP('mlp')
        optimizer_load2 = SGDOptimizer(
                learning_rate=fluid.layers.natural_exp_decay(
                learning_rate=0.1,
                decay_steps=10000,
                decay_rate=0.5,
                staircase=True))
        parameters, optimizers = fluid.dygraph.load_persistables(
            "save_dir_2")
        mlp_load.load_dict(parameters)
        optimizer_load2.load(optimizers)
    self.assertTrue(optimizer2._learning_rate.__dict__ == optimizer_load2._learning_rate.__dict__)


.. py:method:: minimize(loss, startup_program=None, parameter_list=None, no_grad_set=None, grad_clip=None)


通过更新parameter_list来添加操作，进而使损失最小化。

该算子相当于backward()和apply_gradients()功能的合体。

参数：
    - **loss** (Variable) – 用于优化过程的损失值变量
    - **startup_program** (Program) – 用于初始化在parameter_list中参数的startup_program
    - **parameter_list** (list) – 待更新的Variables组成的列表
    - **no_grad_set** (set|None) – 应该被无视的Variables集合
    - **grad_clip** (GradClipBase|None) – 梯度裁剪的策略

返回： (optimize_ops, params_grads)，分别为附加的算子列表；一个由(param, grad) 变量对组成的列表，用于优化

返回类型：   tuple


