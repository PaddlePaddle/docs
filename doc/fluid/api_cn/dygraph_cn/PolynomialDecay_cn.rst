.. _cn_api_fluid_dygraph_PolynomialDecay:

PolynomialDecay
-------------------------------

.. py:class:: paddle.fluid.dygraph.PolynomialDecay(learning_rate, decay_steps, end_learning_rate=0.0001, power=1.0, cycle=False, begin=0, step=1, dtype='float32')

该接口提供学习率按多项式衰减的功能。通过多项式衰减函数，使得学习率值逐步从初始的`learning_rate`，衰减到`end_learning_rate`。

计算方式如下。


.. code-block:: text

    if cycle:
        decay_steps = decay_steps * ceil(global_step / decay_steps)
    else:
        global_step = min(global_step, decay_steps)
        decayed_learning_rate = (learning_rate - end_learning_rate) *
            (1 - global_step / decay_steps) ^ power + end_learning_rate

参数：
    - **learning_rate** (Variable|float32) - 初始学习率。如果设置为Variable，则是标量tensor，数据量类型可以为float32，float64。也可以设置为Python float值。
    - **decay_steps** (int) - 衰减步数。必须是正整数，该参数确定衰减周期。
    - **end_learning_rate** (float) - 最低的最终学习率。
    - **power** (float) - 多项式的幂。 
    - **cycle** (bool) - 学习率下降后是否重新上升。若为True，则学习率衰减到最低学习率值时，会出现上升。若为False，则学习率曲线则单调递减。
    - **begin** (int) – 起始步，即以上运算式子中global_step的初始化值。默认值为0。
    - **step** (int) – 步大小，即以上运算式子中global_step的递增值。默认值为1。
    - **dtype** (str)– 初始化学习率变量的数据类型。默认值为"float32"。


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




