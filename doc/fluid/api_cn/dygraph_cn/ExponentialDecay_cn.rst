.. _cn_api_fluid_dygraph_ExponentialDecay:

ExponentialDecay
-------------------------------

.. py:class:: paddle.fluid.dygraph.ExponentialDecay(learning_rate, decay_steps, decay_rate, staircase=False, begin=0, step=1, dtype='float32')

对学习率应用指数衰减。

在学习率上运用指数衰减。
训练模型时，推荐在训练过程中降低学习率。每次 ``decay_steps`` 步骤中用 ``decay_rate`` 衰减学习率。

.. code-block:: text

    if staircase == True:
        decayed_learning_rate = learning_rate * decay_rate ^ floor(global_step / decay_steps)
    else:
        decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)

参数：
    - **learning_rate** (Variable|float)-初始学习率
    - **decay_steps** (int)-见以上衰减运算
    - **decay_rate** (float)-衰减率。见以上衰减运算
    - **staircase** (Boolean)-若为True,按离散区间衰减学习率。默认：False
    - **begin** (int) - 起始步，默认为0。
    - **step** (int) - 步大小，默认为1。
    - **dtype**  (str) - 学习率的dtype，默认为‘float32’


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







