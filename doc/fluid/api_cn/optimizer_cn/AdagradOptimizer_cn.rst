.. _cn_api_fluid_optimizer_AdagradOptimizer:

AdagradOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.AdagradOptimizer(learning_rate, epsilon=1e-06, regularization=None, name=None, initial_accumulator_value=0.0)

Adaptive Gradient 优化器(自适应梯度优化器，简称Adagrad)可以针对不同参数样本数不平均的问题，自适应地为各个参数分配不同的学习率。

其参数更新的计算过程如下：

.. math::

    moment\_out &= moment + grad * grad\\param\_out 
    &= param - \frac{learning\_rate * grad}{\sqrt{moment\_out} + \epsilon}


相关论文：`Adaptive Subgradient Methods for Online Learning and Stochastic Optimization <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_。

原始论文的算法中没有引入上述公式中的 ``epsilon`` 属性，此处引入该属性用于维持数值稳定性，避免除0错误发生。

引入epsilon参数依据：`Per-parameter adaptive learning rate methods <http://cs231n.github.io/neural-networks-3/#ada>`_。

参数：
    - **learning_rate** (float|Variable) - 学习率，用于参数更新的计算。可以是一个浮点型值或者一个值为浮点型的Variable
    - **epsilon** (float, 可选) - 维持数值稳定性的浮点型值，默认值为1e-06
    - **parameter_list** (list, 可选) - 指定优化器需要优化的参数。在动态图模式下必须提供该参数；在静态图模式下默认值为None，这时所有的参数都将被优化。
    - **regularization** (WeightDecayRegularizer, 可选) - 正则化函数，用于减少泛化误差。例如可以是 :ref:`cn_api_fluid_regularizer_L2DecayRegularizer` ，默认值为None
    - **name** (str, 可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None
    - **initial_accumulator_value** (float, 可选) - moment累加器的初始值，默认值为0.0

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

为网络添加反向计算过程，并根据反向计算所得的梯度，更新parameter_list中的Parameters，最小化网络损失值loss。

参数：
    - **loss** (Variable) – 需要最小化的损失值变量
    - **startup_program** (Program, 可选) – 用于初始化parameter_list中参数的 :ref:`cn_api_fluid_Program` , 默认值为None，此时将使用 :ref:`cn_api_fluid_default_startup_program` 
    - **parameter_list** (list, 可选) – 待更新的Parameter组成的列表， 默认值为None，此时将更新所有的Parameter
    - **no_grad_set** (set, 可选) – 不需要更新的Parameter的集合，默认值为None
    - **grad_clip** (GradClipBase, 可选) – 梯度裁剪的策略，静态图模式不需要使用本参数，当前本参数只支持在dygraph模式下的梯度裁剪，未来本参数可能会调整，默认值为None

返回： (optimize_ops, params_grads)，数据类型为(list, list)，其中optimize_ops是minimize接口为网络添加的OP列表，params_grads是一个由(param, grad)变量对组成的列表，param是Parameter，grad是该Parameter对应的梯度值

返回类型： tuple

**代码示例**

.. code-block:: python

    import numpy as np
    import paddle.fluid as fluid
     
    inp = fluid.layers.data(
        name="inp", shape=[2, 2], append_batch_size=False)
    out = fluid.layers.fc(inp, size=3)
    out = fluid.layers.reduce_sum(out)
    optimizer = fluid.optimizer.AdagradOptimizer(learning_rate=0.2)
    optimizer.minimize(out)

    np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    exe.run(
        feed={"inp": np_inp},
        fetch_list=[out.name])


.. py:method:: clear_gradients()

**注意：**

  **1. 该API只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**


清除需要优化的参数的梯度。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np

    with fluid.dygraph.guard():
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = fluid.dygraph.to_variable(value)
        linear = fluid.Linear(13, 5, dtype="float32")
        optimizer = fluid.optimizer.AdagradOptimizer(learning_rate=0.2,
                                                     parameter_list=linear.parameters())
        out = linear(a)
        out.backward()
        optimizer.minimize(out)
        optimizer.clear_gradients()

