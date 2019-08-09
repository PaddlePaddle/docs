.. _cn_api_fluid_optimizer_LambOptimizer:

LambOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.LambOptimizer(learning_rate=0.001, lamb_weight_decay=0.01, beta1=0.9, beta2=0.999, epsilon=1e-06, regularization=None, exclude_from_weight_decay_fn=None, name=None)

LAMB（Layer-wise Adaptive Moments optimizer for Batching training）优化器
LAMB优化器旨在不降低准确性的条件下扩大训练的批量大小，支持自适应元素更新和精确的分层校正。 更多信息请参考 `Large Batch Optimization for
Deep Learning: Training BERT in 76 minutes <https://arxiv.org/pdf/1904.00962.pdf>`_ 。
参数更新如下：

.. math::

    \begin{align}\begin{aligned}m_t &= \beta_1 m_{t - 1}+ (1 - \beta_1)g_t \\\v_t &= \beta_2 v_{t - 1}  + (1 - \beta_2)g_t^2 \\\r_t &= \frac{m_t}{\sqrt{v_t}+\epsilon} \\\w_t &= w_{t-1} -\eta_t \frac{\left \| w_{t-1}\right \|}{\left \| r_t + \lambda w_{t-1}\right \|} (r_t + \lambda w_{t-1})\end{aligned}\end{align}

其中 :math:`m` 为第一个时刻，:math:`v` 为第二个时刻，:math:`\eta` 为学习率，:math:`\lambda` 为LAMB权重衰减率。

参数：
    - **learning_rate** (float|Variable) – 用于更新参数的学习速率。可以是浮点值或具有一个作为数据元素的浮点值的变量。
    - **lamb_weight_decay** (float) – LAMB权重衰减率。
    - **beta1** (float) – 第一个时刻估计的指数衰减率。
    - **beta2** (float) – 第二个时刻估计的指数衰减率。
    - **epsilon** (float) – 一个小的浮点值，目的是维持数值稳定性。
    - **regularization** (Regularizer) – 一个正则化器，如fluid.regularizer.L1DecayRegularizer。
    - **exclude_from_weight_decay_fn** (function) – 当返回值为True时从权重衰减中去除某个参数。 
    - **name** (str|None) – 名字前缀（可选项）。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
     
    data = fluid.layers.data(name='x', shape=[5], dtype='float32')
    hidden = fluid.layers.fc(input=data, size=10)
    cost = fluid.layers.mean(hidden)

    def exclude_fn(param):
        return param.name.endswith('.b_0')
     
    optimizer = fluid.optimizer.Lamb(learning_rate=0.002,
                                     exclude_from_weight_decay_fn=exclude_fn)
    optimizer.minimize(cost)


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
