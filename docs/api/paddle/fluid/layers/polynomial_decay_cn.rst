.. _cn_api_fluid_layers_polynomial_decay:

polynomial_decay
-------------------------------

.. py:function:: paddle.fluid.layers.polynomial_decay(learning_rate,decay_steps,end_learning_rate=0.0001,power=1.0,cycle=False)




对初始学习率使用多项式衰减

.. code-block:: text

    if cycle:
        decay_steps = decay_steps * ceil(global_step / decay_steps)
    else:
        global_step = min(global_step, decay_steps)
        decayed_learning_rate = (learning_rate - end_learning_rate) *
            (1 - global_step / decay_steps) ^ power + end_learning_rate

参数
::::::::::::

    - **learning_rate** (Variable|float) - 训练过程中的初始学习率，数据类型为float的常数或变量。
    - **decay_steps** (int) - 衰减步数
    - **end_learning_rate** (float) - 训练过程的最终学习率
    - **power** (float) - 多项式衰减系数
    - **cycle** (bool) - step 超出 decay_steps 后是否继续循环，默认为False

返回
::::::::::::
衰减的学习率

返回类型
::::::::::::
变量（Variable）

代码示例
::::::::::::

.. code-block:: python

        import paddle.fluid as fluid
        start_lr = 0.01
        total_step = 5000
        end_lr = 0
        lr = fluid.layers.polynomial_decay(
            start_lr, total_step, end_lr, power=1)








