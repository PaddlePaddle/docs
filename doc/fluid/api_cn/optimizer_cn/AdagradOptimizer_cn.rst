.. _cn_api_fluid_optimizer_AdagradOptimizer:

AdagradOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.AdagradOptimizer(learning_rate, epsilon=1e-06, regularization=None, name=None, initial_accumulator_value=0.0)

**Adaptive Gradient Algorithm(自适应梯度算法，简称Adagrad)**

该算法的参数更新计算过程如下：

.. math::

    moment\_out &= moment + grad * grad\\param\_out 
    &= param - \frac{learning\_rate * grad}{\sqrt{moment\_out} + \epsilon}

原始论文的算法中没有引入上述公式中的epsilon属性，此处引入该属性用于维持数值稳定性，避免除0错误发生。

原始论文地址：`Adaptive Subgradient Methods for Online Learning and Stochastic Optimization <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_。

引入epsilon参数依据：`Per-parameter adaptive learning rate methods <http://cs231n.github.io/neural-networks-3/#ada>`_。

参数：
    - **learning_rate** (float|Variable) - 学习率，用于参数更新的计算。可以是一个浮点型值或者一个值为浮点型的Variable。
    - **epsilon** (float) - 维持数值稳定性的浮点型值，默认值为1e-06
    - **regularization (function|None)** - 正则化函数，用于减少泛化误差。例如可以是fluid.regularizer.L2DecayRegularizer，默认值为None。
    - **name (str|None)** - 该参数供开发人员打印调试信息时使用，具体用法请参见 Name，默认值为None。
    - **initial_accumulator_value** (float) - moment累加器的初始值，默认值为0.0。

**代码示例**

.. code-block:: python

    import numpy as np
    import paddle.fluid as fluid
     
    np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    inp = fluid.layers.data(
        name="inp", shape=[2, 2], append_batch_size=False)
    out = fluid.layers.fc(inp, size=3)
    out = fluid.layers.reduce_sum(out)
    optimizer = fluid.optimizer.AdagradOptimizer(learning_rate=0.2)
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
    - **loss** (Variable) – 用于优化过程的损失值变量
    - **startup_program** (Program) – 用于初始化在parameter_list中参数的startup_program
    - **parameter_list** (list) – 待更新的Variables组成的列表
    - **no_grad_set** (set|None) – 应该被无视的Variables集合
    - **grad_clip** (GradClipBase|None) – 梯度裁剪的策略

返回： (optimize_ops, params_grads)，分别为附加的算子列表；一个由(param, grad) 变量对组成的列表，用于优化

返回类型：   tuple








