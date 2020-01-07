.. _cn_api_fluid_optimizer_LarsMomentumOptimizer:

LarsMomentumOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.LarsMomentumOptimizer(learning_rate, momentum, lars_coeff=0.001, lars_weight_decay=0.0005, regularization=None, name=None)

该接口实现LARS支持的Momentum优化器

公式作如下更新：

.. math::

  & local\_learning\_rate = learning\_rate * lars\_coeff * \
  \frac{||param||}{||gradient|| + lars\_weight\_decay * ||param||}\\
  & velocity = mu * velocity + local\_learning\_rate * (gradient + lars\_weight\_decay * param)\\
  & param = param - velocity

参数：
  - **learning_rate** (float|Variable) - 学习率，用于参数更新。作为数据参数，可以是浮点型值或含有一个浮点型值的变量。
  - **momentum** (float) - 动量因子。
  - **parameter_list** (list, 可选) - 指定优化器需要优化的参数。在动态图模式下必须提供该参数；在静态图模式下默认值为None，这时所有的参数都将被优化。
  - **lars_coeff** (float，可选) - 定义LARS本地学习率的权重，默认值0.001。
  - **lars_weight_decay** (float，可选) - 使用LARS进行衰减的权重衰减系数，默认值0.0005。
  - **regularization** - 正则化函数，例如 :code:`fluid.regularizer.L2DecayRegularizer`。
  - **name** (str, 可选) - 可选的名称前缀，一般无需设置，默认值为None。


**代码示例**

.. code-block:: python

    import paddle.fluid as fluid

    np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    inp = fluid.layers.data(
        name="inp", shape=[2, 2], append_batch_size=False)
    out = fluid.layers.fc(inp, size=3)
    out = fluid.layers.reduce_sum(out)
    optimizer = fluid.optimizer.LarsMomentumOptimizer(learning_rate=0.001, momentum=0.9)
    optimizer.minimize(out)

    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    exe.run(
        feed={"inp": np_inp},
        fetch_list=[out.name])



.. py:method:: minimize(loss, startup_program=None, parameter_list=None, no_grad_set=None, grad_clip=None)


通过更新parameter_list来添加操作，进而使损失最小化。

该算子相当于backward()和apply_gradients()功能的合体。

参数：
    - **loss** (Variable) – 用于优化过程的损失值变量。
    - **startup_program** (Program) – 用于初始化在parameter_list中参数的startup_program。
    - **parameter_list** (list) – 待更新的Variables组成的列表。
    - **no_grad_set** (set|None) – 应该被无视的Variables集合。
    - **grad_clip** (GradClipBase|None) – 梯度裁剪的策略。

返回： (optimize_ops, params_grads)，数据类型为(list, list)，其中optimize_ops是minimize接口为网络添加的OP列表，params_grads是一个由(param, grad)变量对组成的列表，param是Parameter，grad是该Parameter对应的梯度值

返回类型： tuple


.. py:method:: clear_gradients()

该函数仅在动态图模式下使用。

清除需要优化的参数的梯度。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np

    with fluid.dygraph.guard():
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = fluid.dygraph.to_variable(value)
        linear = fluid.Linear(13, 5, dtype="float32")
        optimizer = fluid.optimizer.LarsMomentumOptimizer(learning_rate=0.001, momentum=0.9,
                                      parameter_list=linear.parameters())
        out = linear(a)
        out.backward()
        optimizer.minimize(out)
        optimizer.clear_gradients()

