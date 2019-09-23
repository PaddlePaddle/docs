.. _cn_api_fluid_optimizer_AdamaxOptimizer:

AdamaxOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.AdamaxOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, regularization=None, name=None)

我们参考Adam论文第7节中的Adamax优化: https://arxiv.org/abs/1412.6980 ， Adamax是基于无穷大范数的Adam算法的一个变种。


Adamax 更新规则:

.. math::
    \\t = t + 1
.. math::
    moment\_out=\beta_1∗moment+(1−\beta_1)∗grad
.. math::
    inf\_norm\_out=\max{(\beta_2∗inf\_norm+ϵ, \left|grad\right|)}
.. math::
    learning\_rate=\frac{learning\_rate}{1-\beta_1^t}
.. math::
    param\_out=param−learning\_rate*\frac{moment\_out}{inf\_norm\_out}\\


论文中没有 ``epsilon`` 参数。但是，为了数值稳定性， 防止除0错误， 增加了这个参数

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import numpy
     
    # First create the Executor.
    place = fluid.CPUPlace() # fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
     
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(train_program, startup_program):
        data = fluid.layers.data(name='X', shape=[1], dtype='float32')
        hidden = fluid.layers.fc(input=data, size=10)
        loss = fluid.layers.mean(hidden)
        adam = fluid.optimizer.Adamax(learning_rate=0.2)
        adam.minimize(loss)
     
    # Run the startup program once and only once.
    exe.run(startup_program)
     
    x = numpy.random.random(size=(10, 1)).astype('float32')
    outs = exe.run(program=train_program,
                  feed={'X': x},
                   fetch_list=[loss.name])

参数:
  - **learning_rate**  (float|Variable) - 用于更新参数的学习率。可以是浮点值，也可以是具有一个浮点值作为数据元素的变量。
  - **beta1** (float) - 第1阶段估计的指数衰减率
  - **beta2** (float) - 第2阶段估计的指数衰减率。
  - **epsilon** (float) -非常小的浮点值，为了数值的稳定性质
  - **regularization** - 正则化器，例如 ``fluid.regularizer.L2DecayRegularizer`` 
  - **name** - 可选的名称前缀。

.. note::
    目前 ``AdamaxOptimizer`` 不支持  sparse parameter optimization.



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

为网络添加反向计算过程，并根据反向计算所得的梯度，更新parameter_list中的Parameters，最小化网络损失值loss。

参数：
    - **loss** (Variable) – 需要最小化的损失值变量
    - **startup_program** (Program, 可选) – 用于初始化parameter_list中参数的 :ref:`cn_api_fluid_Program` , 默认值为None，此时将使用 :ref:`cn_api_fluid_default_startup_program` 
    - **parameter_list** (list, 可选) – 待更新的Parameter组成的列表， 默认值为None，此时将更新所有的Parameter
    - **no_grad_set** (set, 可选) – 不需要更新的Parameter的集合，默认值为None
    - **grad_clip** (GradClipBase, 可选) – 梯度裁剪的策略，默认值为None

返回： (optimize_ops, params_grads)，数据类型为(list, list)，其中optimize_ops是minimize接口为网络添加的OP列表，params_grads是一个由(param, grad)变量对组成的列表，param是Parameter，grad是该Parameter对应的梯度值

**代码示例**：

.. code-block:: python

    import numpy
    import paddle.fluid as fluid
     
    data = fluid.layers.data(name='X', shape=[1], dtype='float32')
    hidden = fluid.layers.fc(input=data, size=10)
    loss = fluid.layers.mean(hidden)
    adam = fluid.optimizer.Adamax(learning_rate=0.2)
    adam.minimize(loss)

    place = fluid.CPUPlace() # fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
     
    x = numpy.random.random(size=(10, 1)).astype('float32')
    exe.run(fluid.default_startup_program())
    outs = exe.run(program=fluid.default_main_program(),
                   feed={'X': x},
                   fetch_list=[loss.name])








