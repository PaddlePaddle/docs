.. _cn_api_fluid_layers_inverse_time_decay:

inverse_time_decay
-------------------------------

.. py:function:: paddle.fluid.layers.inverse_time_decay(learning_rate, decay_steps, decay_rate, staircase=False)




在初始学习率上运用逆时衰减。

训练模型时，最好在训练过程中降低学习率。通过执行该函数，将对初始学习率运用逆时衰减函数。

逆时衰减计算方式如下。

.. code-block:: python

    if staircase == True:
         decayed_learning_rate = learning_rate / (1 + decay_rate * floor(global_step / decay_step))
     else:
         decayed_learning_rate = learning_rate / (1 + decay_rate * global_step / decay_step)

参数
::::::::::::

    - **learning_rate** (Variable|float) - 初始学习率，类型可以为学习率变量(Variable)或float型常量。
    - **decay_steps** (int) - 学习率衰减步长，见以上衰减运算。
    - **decay_rate** (float) - 学习率衰减率。见以上衰减运算。
    - **staircase** (bool) - 若为True，按离散区间衰减学习率，即每 ``decay_steps`` 步多衰减 ``decay_rate`` 倍。若为False，则按以上衰减运算持续衰减。默认False。

返回
::::::::::::
Variable(Tensor) 随step衰减的学习率变量，维度为 :math:`[1]` 的1-D Tensor。

返回类型
::::::::::::
变量(Variable)

代码示例
::::::::::::

.. code-block:: python

        import paddle.fluid as fluid
        base_lr = 0.1
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.layers.natural_exp_decay(
                learning_rate=base_lr,
                decay_steps=10000,
                decay_rate=0.5,
                staircase=True))





