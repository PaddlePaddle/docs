.. _cn_api_fluid_dygraph_ExponentialDecay:

ExponentialDecay
-------------------------------

.. py:class:: paddle.fluid.dygraph.ExponentialDecay(learning_rate, decay_steps, decay_rate, staircase=False, begin=0, step=1, dtype=’float32‘)

该接口提供一种学习率按指数函数衰减的功能。

计算方式如下。

.. code-block:: text

    if staircase == True:
        decayed_learning_rate = learning_rate * decay_rate ^ floor(global_step / decay_steps)
    else:
        decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)

参数：
    - **learning_rate** (Variable|float) - 初始学习率。如果设置为Variable，则是标量Tensor，数据类型可以为float32，float64。也可以设置为Python float值。
    - **decay_steps** (int) - 衰减步数。必须是正整数，该参数确定衰减周期。
    - **decay_rate** (float)- 衰减率。
    - **staircase** (bool) - 若为True，则以不连续的间隔衰减学习速率即阶梯型衰减。若为False，则以标准指数型衰减。默认值为False。
    - **begin** (int) - 起始步，即以上运算式子中global_step的初始化值。默认值为0。
    - **step** (int) - 步大小，即以上运算式子中global_step的每次的增量值，使得global_step随着训练的次数递增。默认值为1。
    - **dtype** (str) - 初始化学习率变量的数据类型。默认值为"float32"。


**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    base_lr = 0.1
    with fluid.dygraph.guard():
        sgd_optimizer = fluid.optimizer.SGD(
              learning_rate=fluid.dygraph.ExponentialDecay(
                  learning_rate=base_lr,
                  decay_steps=10000,
                  decay_rate=0.5,
                  staircase=True))







