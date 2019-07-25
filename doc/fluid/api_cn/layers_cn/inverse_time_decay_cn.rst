.. _cn_api_fluid_layers_inverse_time_decay:

inverse_time_decay
-------------------------------

.. py:function:: paddle.fluid.layers.inverse_time_decay(learning_rate, decay_steps, decay_rate, staircase=False)

在初始学习率上运用逆时衰减。

训练模型时，最好在训练过程中降低学习率。通过执行该函数，将对初始学习率运用逆向衰减函数。

.. code-block:: python

    if staircase == True:
         decayed_learning_rate = learning_rate / (1 + decay_rate * floor(global_step / decay_step))
     else:
         decayed_learning_rate = learning_rate / (1 + decay_rate * global_step / decay_step)

参数：
    - **learning_rate** (Variable|float)-初始学习率
    - **decay_steps** (int)-见以上衰减运算
    - **decay_rate** (float)-衰减率。见以上衰减运算
    - **staircase** (Boolean)-若为True，按间隔区间衰减学习率。默认：False

返回：衰减的学习率

返回类型：变量（Variable）

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
        sgd_optimizer.minimize(avg_cost)




