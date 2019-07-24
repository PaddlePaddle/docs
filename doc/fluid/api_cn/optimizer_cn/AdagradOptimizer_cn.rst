.. _cn_api_fluid_optimizer_AdagradOptimizer:

AdagradOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.AdagradOptimizer(learning_rate, epsilon=1e-06, regularization=None, name=None, initial_accumulator_value=0.0)

**Adaptive Gradient Algorithm(Adagrad)**

更新如下：

.. math::

    moment\_out &= moment + grad * grad\\param\_out 
    &= param - \frac{learning\_rate * grad}{\sqrt{moment\_out} + \epsilon}

原始论文（http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf）没有epsilon属性。在我们的实现中也作了如下更新：
http://cs231n.github.io/neural-networks-3/#ada 用于维持数值稳定性，避免除数为0的错误发生。

参数：
    - **learning_rate** (float|Variable)-学习率，用于更新参数。作为数据参数，可以是一个浮点类型值或者有一个浮点类型值的变量
    - **epsilon** (float) - 维持数值稳定性的短浮点型值
    - **regularization** - 规则化函数，例如fluid.regularizer.L2DecayRegularizer
    - **name** - 名称前缀（可选）
    - **initial_accumulator_value** (float) - moment累加器的初始值。

**代码示例**：

.. code-block:: python:

    import paddle.fluid as fluid
    import numpy as np
     
    np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    inp = fluid.layers.data(
        name="inp", shape=[2, 2], append_batch_size=False)
    out = fluid.layers.fc(inp, size=3)
    out = fluid.layers.reduce_sum(out)
    optimizer = fluid.optimizer.Adagrad(learning_rate=0.2)
    optimizer.minimize(out)

    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    exe.run(
        feed={"inp": np_inp},
        fetch_list=[out.name])






