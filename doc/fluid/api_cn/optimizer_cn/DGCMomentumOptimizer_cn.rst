.. _cn_api_fluid_optimizer_DGCMomentumOptimizer:

DGCMomentumOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.DGCMomentumOptimizer(learning_rate, momentum, rampup_begin_step, rampup_step=1, sparsity=[0.999], use_nesterov=False, local_grad_clip_norm=None, num_trainers=None, regularization=None, name=None)

DGC（深度梯度压缩）Momentum 优化器。原始论文: https://arxiv.org/abs/1712.01887

DGC通过只传送重要梯度（稀疏更新）的方式，即只发送大于给定阈值的梯度，来减少通信带宽使用。

DGC会在本地累加剩余梯度以避免信息的丢失。最终这些梯度会大到足以传输。

因此，DGC只会立即发送大梯度，但随时间流逝所有梯度终将发送出去。

为确保精度不会损失，DGC在梯度稀疏化之上采用动量修正和局部梯度修剪(clip)来维持模型性能。

DGC还使用动量因子掩藏（momentum factor masking）和预训练（warm-up）来克服由于规约（reduced）通信而导致的数据陈旧性（staleness）问题。

这个优化器会执行如下操作：

1. 从张量中获取的前TopK个重要梯度进行压缩，并将其用于allreduce通信以减少网络带宽使用。
2. 调用momentum来优化代价函数。

参数: 
    - **learning_rate** （float | Variable） - 用于更新参数的学习率。可以是浮点值或由一个浮点型数据组成的Variable。
    - **momentum** （float） - 动量因子。
    - **rampup_begin_step** （int） - 进行梯度压缩的起步点。
    - **rampup_step** （int） - 使用稀疏预热的时间步长。默认值为1。例如：如果稀疏度为[0.75,0.9375,0.984375,0.996,0.999]，并且rampup_step为100，则在0~19步时使用0.75，在20~39步时使用0.9375，依此类推。当到达sparsity数组末尾时，此后将会使用0.999。
    - **sparsity** （list [float]） - 从梯度张量中获取top个重要元素，比率为（1-当前稀疏度）。默认值为[0.999]。例如：如果sparsity为[0.99, 0.999]，则将传输top [1%, 0.1%]的重要元素。
    - **use_nesterov** （bool） - 启用Nesterov momentum。 True意味着使用Nesterov。默认值False。
    - **local_grad_clip_norm** （float，可选） - 局部梯度裁减标准值。可选，默认为None，表示不需要裁减。
    - **num_trainers** （int，可选） - 训练节点的数量。可选，默认为None。
    - **regularization** （WeightDecayRegularizer，可选） - 正则器， 如 :ref:`cn_api_fluid_regularizer_L2DecayRegularizer`。可选，默认为None。
    - **name** （str，可选） - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    optimizer = fluid.optimizer.DGCMomentumOptimizer(
                                        learning_rate=0.0001,
                                        momentum=0.9,
                                        rampup_step=1000,
                                        rampup_begin_step=1252,
                                        sparsity=[0.999, 0.999])




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

