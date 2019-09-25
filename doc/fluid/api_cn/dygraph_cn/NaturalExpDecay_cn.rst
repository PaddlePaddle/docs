.. _cn_api_fluid_dygraph_NaturalExpDecay:

NaturalExpDecay
-------------------------------

.. py:class:: paddle.fluid.dygraph.NaturalExpDecay(learning_rate, decay_steps, decay_rate, staircase=False, begin=0, step=1, dtype='float32')

该接口为优化器提供学习率按指数衰减的功能。

计算方式如下。

.. code-block:: text

    if not staircase:
        decayed_learning_rate = learning_rate * exp(- decay_rate * (global_step / decay_steps))
    else:
        decayed_learning_rate = learning_rate * exp(- decay_rate * floor(global_step / decay_steps))

参数：
    - **learning_rate** (Variable|float) - 初始学习率值。如果设置为Variable，则是标量tensor，数据量类型可以为float32，float64。也可以设置为Python float值。
    - **decay_steps** (int) – 指定衰减的步长。该参数确定衰减的周期。
    - **decay_rate** (float) – 指定衰减率。
    - **staircase** (bool) - 若为True, 学习率变化曲线呈阶梯状，若为False，学习率变化值曲线为平滑的曲线。默认值为False。
    - **begin** (int) – 起始步，即以上运算式子中global_step的初始化值。默认值为0。
    - **step** (int) – 步大小，即以上运算式子中global_step的每次的增量值。默认值为1。
    - **dtype**  – (str) 初始化学习率变量的数据类型。默认值为"float32"。


**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    base_lr = 0.1
    with fluid.dygraph.guard():
        sgd_optimizer = fluid.optimizer.SGD(
                learning_rate=fluid.dygraph.NaturalExpDecay(
                      learning_rate=base_lr,
                      decay_steps=10000,
                      decay_rate=0.5,
                      staircase=True))





