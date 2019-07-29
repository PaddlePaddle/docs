.. _cn_api_fluid_dygraph_PolynomialDecay:

PolynomialDecay
-------------------------------

.. py:class:: paddle.fluid.dygraph.PolynomialDecay(learning_rate, decay_steps, end_learning_rate=0.0001, power=1.0, cycle=False, begin=0, step=1, dtype='float32')

为初始学习率应用多项式衰减。


.. code-block:: text

    if cycle:
        decay_steps = decay_steps * ceil(global_step / decay_steps)
    else:
        global_step = min(global_step, decay_steps)
        decayed_learning_rate = (learning_rate - end_learning_rate) *
            (1 - global_step / decay_steps) ^ power + end_learning_rate

参数：
    - **learning_rate** (Variable|float32)-标量float32值或变量。是训练过程中的初始学习率
    - **decay_steps** (int32)-Python int32数
    - **end_learning_rate** (float)-Python float数
    - **power** (float)-Python float数
    - **cycle** (bool)-若设为true，每decay_steps衰减学习率
    - **begin** (int) – 起始步(默认为0)
    - **step** (int) – 步大小(默认为1)
    - **dtype**  (str)– 初始化学习率变量的dtype(默认为‘float32’)

返回：衰减的学习率

返回类型：变量（Variable）

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    start_lr = 0.01
    total_step = 5000
    end_lr = 0
    with fluid.dygraph.guard():
        optimizer  = fluid.optimizer.SGD(
            learning_rate = fluid.dygraph.PolynomialDecay(
            start_lr, total_step, end_lr, power=1.0) )




