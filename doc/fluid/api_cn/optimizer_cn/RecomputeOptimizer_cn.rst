.. _cn_api_fluid_optimizer_RecomputeOptimizer:

RecomputeOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.RecomputeOptimizer(optimizer)

通常来讲，一个深度学习的训练流程包含了三个子步骤：首先，运行前向算子来计算Variable和loss的值；其次，运行反向算子来计算参数的梯度；最后，应用优化算法以更新参数值。

在前向运算过程中，反向运算会用到的Variable都会保存在内存中，当模型深度很深时，这会占用大量的内存。

重计算将深度学习网络切分为k个部分（segments）。在每个segment，运行反向运算时会首先运算前向计算。这对节省显存非常有益。

把一个深度学习网络切分为k个segments的Variables被称为checkpoints。用户在使用运行RecomputeOptimizer之前需要先设置checkpoints。

参数: 
    - **optimizer** (Optimizer)-内部优化器

**代码示例**：

.. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            def gen_data():
                return {"x": np.random.random(size=(32, 32)).astype('float32'),
                "y": np.random.randint(2, size=(32, 1)).astype('int64')}
            def mlp(input_x, input_y, hid_dim=128, label_dim=2):
                print(input_x)
                fc_1 = fluid.layers.fc(input=input_x, size=hid_dim)
                prediction = fluid.layers.fc(input=[fc_1], size=label_dim, act='softmax')
                cost = fluid.layers.cross_entropy(input=prediction, label=input_y)
                sum_cost = fluid.layers.reduce_mean(cost)
                return sum_cost, fc_1, prediction
            input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
            input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')
            cost, fc_1, pred = mlp(input_x, input_y)

            sgd = fluid.optimizer.Adam(learning_rate=0.01)
            sgd = fluid.optimizer.RecomputeOptimizer(sgd)
            sgd._set_checkpoints([fc_1, pred])
            sgd.minimize(cost)

            print("Finished optimize")
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            step = 10

            for i in range(step):
                cost_val = exe.run(feed=gen_data(),
                       program=fluid.default_main_program(),
                       fetch_list=[cost.name])
                print("step=%d cost=%f" % (i, cost_val[0]))


.. py:method:: apply_gradients(params_grads)

调用self.apply_gradients

参数：
    - **params_grads** (list)- 用于优化的(param, grad)对组成的列表

返回：  附加在当前Program的算子组成的列表

返回类型：  list

**代码示例**

.. code-block:: python

                import paddle.fluid as fluid
                import paddle.fluid.framework as framework

                def mlp(input_x, input_y, hid_dim=128, label_dim=2):
                    fc_1 = fluid.layers.fc(input=input_x, size=hid_dim)
                    prediction = fluid.layers.fc(input=[fc_1], size=label_dim, act='softmax')
                    cost = fluid.layers.cross_entropy(input=prediction, label=input_y)
                    sum_cost = fluid.layers.reduce_mean(cost)
                    return sum_cost, fc_1, prediction


                input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
                input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')
                cost, fc_1, pred = mlp(input_x, input_y)
                print("Finished FF")

                sgd = fluid.optimizer.Adam(learning_rate=0.01)
                sgd = fluid.optimizer.RecomputeOptimizer(sgd)
                params_grads = sgd.backward(
                    cost,
                    startup_program=None,
                    parameter_list=None,
                    no_grad_set=None,
                    checkpoints=[fc_1, pred])

                program = cost.block.program
                with framework.program_guard(program, None):
                    optimize_ops = sgd.apply_gradients(params_grads)

                print("Finished apply gradients")

.. py:method:: apply_optimize(loss, startup_program, params_grads)

调用self._optimizer的apply_optimize函数

参数：
    - **loss** (Variable) – 用于优化过程的损失值变量
    - **startup_program** (Program) – 用于初始化在parameter_list中参数的startup_program
    - **params_grads** (list)- 用于优化的(param, grad)对组成的列表

返回：  附加在当前Program的算子组成的列表

返回类型：  list

**代码示例**

.. code-block:: python

                import paddle.fluid as fluid

                def mlp(input_x, input_y, hid_dim=128, label_dim=2):
                    fc_1 = fluid.layers.fc(input=input_x, size=hid_dim)
                    prediction = fluid.layers.fc(input=[fc_1], size=label_dim, act='softmax')
                    cost = fluid.layers.cross_entropy(input=prediction, label=input_y)
                    sum_cost = fluid.layers.reduce_mean(cost)
                    return sum_cost, fc_1, prediction

                input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
                input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')
                cost, fc_1, pred = mlp(input_x, input_y)
                print("Finished FF")

                sgd = fluid.optimizer.Adam(learning_rate=0.01)
                sgd = fluid.optimizer.RecomputeOptimizer(sgd)
                params_grads = sgd.backward(
                    cost,
                    startup_program=None,
                    parameter_list=None,
                    no_grad_set=None,
                    checkpoints=[fc_1, pred])

                optimize_ops = sgd.apply_optimize(
                    cost, startup_program=None, params_grads=params_grads)

                print("Finished apply_optimize")

.. py:method:: backward(loss, startup_program=None, parameter_list=None, no_grad_set=None, callbacks=None)

带checkpoint的backward函数

参数：
    - **loss** (Variable) – 用于优化过程的损失值变量
    - **startup_program** (Program) – 用于初始化在parameter_list中参数的startup_program
    - **parameter_list** (list) – 待更新的Variables组成的列表
    - **no_grad_set** (set|None) – 应该被无视的Variables集合
    - **callbacks** (list|None) – 当为某参数附加反向算子时所要运行的callables组成的列表
    - **checkpoints** (list|None) – 一批作为checkpoints的Variables

返回：  附加在当前Program的算子组成的列表

返回类型：  list

**代码示例**

.. code-block:: python

                import paddle.fluid as fluid

                def mlp(input_x, input_y, hid_dim=128, label_dim=2):
                    fc_1 = fluid.layers.fc(input=input_x, size=hid_dim)
                    prediction = fluid.layers.fc(input=[fc_1], size=label_dim, act='softmax')
                    cost = fluid.layers.cross_entropy(input=prediction, label=input_y)
                    sum_cost = fluid.layers.reduce_mean(cost)
                    return sum_cost, fc_1, prediction


                input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
                input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')
                cost, fc_1, pred = mlp(input_x, input_y)
                print("Finished FF")

                sgd = fluid.optimizer.Adam(learning_rate=0.01)
                sgd = fluid.optimizer.RecomputeOptimizer(sgd)
                params_grads = sgd.backward(
                    cost,
                    startup_program=None,
                    parameter_list=None,
                    no_grad_set=None,
                    checkpoints=[fc_1, pred])
                print("Finished backward")


.. py:method:: load(stat_dict)

Recompute Optimizer 目前不支持load函数

参数：
    - **stat_dict** – load_persistable方法加载的dict

**代码示例**

.. code-block:: python


                import paddle.fluid as fluid
                import paddle.compat as cpt

                def mlp(input_x, input_y, hid_dim=128, label_dim=2):
                    fc_1 = fluid.layers.fc(input=input_x, size=hid_dim)
                    prediction = fluid.layers.fc(input=[fc_1], size=label_dim, act='softmax')
                    cost = fluid.layers.cross_entropy(input=prediction, label=input_y)
                    sum_cost = fluid.layers.reduce_mean(cost)
                    return sum_cost, fc_1, prediction

                input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
                input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')
                cost, fc_1, pred = mlp(input_x, input_y)
                print("Finished FF")

                sgd = fluid.optimizer.Adam(learning_rate=0.01)
                sgd = fluid.optimizer.RecomputeOptimizer(sgd)
                sgd._set_checkpoints([fc_1, pred])
                try:
                    stat_dict = {}
                    sgd.load(stat_dict)
                except NotImplementedError as e:
                    print(cpt.get_exception_message(e))


