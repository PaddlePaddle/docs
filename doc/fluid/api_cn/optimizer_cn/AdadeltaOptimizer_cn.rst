.. _cn_api_fluid_optimizer_AdadeltaOptimizer:

AdadeltaOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.AdadeltaOptimizer(learning_rate, epsilon=1.0e-6, rho=0.95, regularization=None, name=None)

**Adadelta Optimizer**

包含average squared grad state和average squared update state的Adadelta优化器，具体细节可参考论文 `ADADELTA: AN ADAPTIVE LEARNING RATE METHOD <https://arxiv.org/abs/1212.5701>`_ 。

更新公式如下：

.. math::
    E(g_t^2) &= \rho * E(g_{t-1}^2) + (1-\rho) * g^2
.. math::
    learning\_rate &= \sqrt{ ( E(dx_{t-1}^2) + \epsilon ) / ( E(g_t^2) + \epsilon ) }
.. math::
    E(dx_t^2) &= \rho * E(dx_{t-1}^2) + (1-\rho) * (-g*learning\_rate)^2


参数：
    - **learning_rate** (float|Variable) - 全局学习率
    - **epsilon** (float) - 维持数值稳定性的浮点型值
    - **rho** (float) - 算法中的衰减率
    - **regularization** - 规则化函数，例如fluid.regularizer.L2DecayRegularizer
    - **name** - 名称前缀（可选）

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid

    image = fluid.layers.data(name='image', shape=[28], dtype='float32')
    fc = fluid.layers.fc(image, size=10)
    cost = fluid.layers.reduce_mean(fc)
    optimizer = fluid.optimizer.Adadelta(
        learning_rate=0.0003, epsilon=1.0e-6, rho=0.95)
    _, params_grads = optimizer.minimize(cost)


.. py:method:: apply_gradients(params_grads)

为给定的params_grads对附加优化算子，为minimize过程的第二步

参数：
    - **params_grads** (list)- 用于优化的(param, grad)对组成的列表

返回：  附加在当前Program的算子组成的列表

返回类型：  list

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid

    image = fluid.layers.data(name='image', shape=[28], dtype='float32')
    fc = fluid.layers.fc(image, size=10)
    cost = fluid.layers.reduce_mean(fc)
    optimizer = fluid.optimizer.Adadelta(
        learning_rate=0.0003, epsilon=1.0e-6, rho=0.95)
    params_grads = optimizer.backward(cost)
    optimizer.apply_gradients(params_grads)


.. py:method:: apply_optimize(loss, startup_program, params_grads)

为给定的params_grads对附加优化算子，为minimize过程的第二步。

参数：
    - **loss** (Variable) – 用于优化过程的损失值变量
    - **startup_program** (Program) – 用于初始化在parameter_list中参数的startup_program
    - **params_grads** (list)- 用于优化的(param, grad)对组成的列表

返回：  附加在当前Program的算子组成的列表

返回类型：  list

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid

    image = fluid.layers.data(name='image', shape=[28], dtype='float32')
    fc = fluid.layers.fc(image, size=10)
    cost = fluid.layers.reduce_mean(fc)
    optimizer = fluid.optimizer.Adadelta(
        learning_rate=0.0003, epsilon=1.0e-6, rho=0.95)
    params_grads = optimizer.backward(cost)
    optimizer.apply_optimize(cost, fluid.default_startup_program(), params_grads)


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
    from paddle.fluid.optimizer import *
    from paddle.fluid.dygraph.nn import FC
    from paddle.fluid.dygraph.base import to_variable

    Optimizer = AdadeltaOptimizer

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
        optimizer = Optimizer(learning_rate=fluid.layers.cosine_decay(
            learning_rate=0.1, step_each_epoch=10000, epochs=120))

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
                mlp.state_dict(), "save_dir_2", optimizer)
            if batch_id == 2:
                break

    with fluid.dygraph.guard():
        mlp_load = MLP('mlp')
        optimizer_load = Optimizer(learning_rate=fluid.layers.cosine_decay(
            learning_rate=0.1, step_each_epoch=10000, epochs=120))
        parameters, optimizers = fluid.dygraph.load_persistables(
            "save_dir_2")
        mlp_load.load_dict(parameters)
        optimizer_load.load(optimizers)


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

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid

    image = fluid.layers.data(name='image', shape=[28], dtype='float32')
    fc = fluid.layers.fc(image, size=10)
    cost = fluid.layers.reduce_mean(fc)
    optimizer = fluid.optimizer.Adadelta(
        learning_rate=0.0003, epsilon=1.0e-6, rho=0.95)
    _, params_grads = optimizer.minimize(cost)


