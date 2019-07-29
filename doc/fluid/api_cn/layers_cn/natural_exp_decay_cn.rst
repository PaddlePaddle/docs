.. _cn_api_fluid_layers_natural_exp_decay:

natural_exp_decay
-------------------------------

.. py:function:: paddle.fluid.layers.natural_exp_decay(learning_rate, decay_steps, decay_rate, staircase=False)

将自然指数衰减运用到初始学习率上。

.. code-block:: python

    if not staircase:
        decayed_learning_rate = learning_rate * exp(- decay_rate * (global_step / decay_steps))
    else:
        decayed_learning_rate = learning_rate * exp(- decay_rate * (global_step / decay_steps))

参数：
    - **learning_rate** - 标量float32值或变量。是训练过程中的初始学习率。
    - **decay_steps** - Python int32数
    - **decay_rate** - Python float数
    - **staircase** - Boolean.若设为true，每个decay_steps衰减学习率

返回：衰减的学习率

**示例代码：**

.. code-block:: python

    import paddle.fluid as fluid
    base_lr = 0.1
    sgd_optimizer = fluid.optimizer.SGD(
        learning_rate=fluid.layers.natural_exp_decay(
              learning_rate=base_lr,
              decay_steps=10000,
              decay_rate=0.5,
              staircase=True))






