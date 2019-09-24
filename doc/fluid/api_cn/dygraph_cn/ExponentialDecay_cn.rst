.. _cn_api_fluid_dygraph_ExponentialDecay:

ExponentialDecay
-------------------------------

.. py:class:: paddle.fluid.dygraph.ExponentialDecay(learning_rate, decay_steps, decay_rate, staircase=False, begin=0, step=1, dtype="float32")

该接口提供一种学习率按指数衰减的功能。

在学习率上运用指数衰减。
训练模型时，推荐在训练过程中降低学习率。每次 ``decay_steps`` 步骤中用 ``decay_rate`` 衰减学习率。

.. code-block:: text

    if staircase == True:
        decayed_learning_rate = learning_rate * decay_rate ^ floor(global_step / decay_steps)
    else:
        decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)

参数：
    - **learning_rate** (Variable|float) - 初始学习率。
    - **decay_steps** (int) - 衰减步数。必须是正整数，决定衰减周期。
    - **decay_rate** (float)- 衰减率。
    - **staircase** (bool) - 若为True，则以不连续的间隔衰减学习速率即阶梯型衰减。若为False，则以标准指数型衰减。默认值为False。
    - **begin** (int) - 起始步。默认值为0。
    - **step** (int) - 步大小。默认值为1。
    - **dtype**  (str) - 学习率的数据类型，默认值为"float32"


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







