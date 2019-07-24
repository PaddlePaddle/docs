.. _cn_api_fluid_dygraph_NaturalExpDecay:

NaturalExpDecay
-------------------------------

.. py:class:: paddle.fluid.dygraph.NaturalExpDecay(learning_rate, decay_steps, decay_rate, staircase=False, begin=0, step=1, dtype='float32')

为初始学习率应用指数衰减策略。

.. code-block:: text

    if not staircase:
        decayed_learning_rate = learning_rate * exp(- decay_rate * (global_step / decay_steps))
    else:
        decayed_learning_rate = learning_rate * exp(- decay_rate * (global_step / decay_steps))

参数：
    - **learning_rate** (Variable|float)- 类型为float32的标量值或为一个Variable。它是训练的初始学习率。
    - **decay_steps** (int)-一个Python int32 数。
    - **decay_rate** (float)- 一个Python float数。
    - **staircase** (Boolean)-布尔型。若为True,每隔decay_steps衰减学习率。
    - **begin**  – Python ‘int32’ 数，起始步(默认为0)。
    - **step**  – Python ‘int32’ 数, 步大小(默认为1)。
    - **dtype**  – Python ‘str’ 类型, 初始化学习率变量的dtype(默认为‘float32’)。


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





