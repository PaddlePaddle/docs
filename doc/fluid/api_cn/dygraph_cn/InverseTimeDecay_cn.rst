.. _cn_api_fluid_dygraph_InverseTimeDecay:

InverseTimeDecay
-------------------------------

.. py:class:: paddle.fluid.dygraph.InverseTimeDecay(learning_rate, decay_steps, decay_rate, staircase=False, begin=0, step=1, dtype='float32')

在初始学习率上运用逆时衰减。

训练模型时，最好在训练过程中降低学习率。通过执行该函数，将对初始学习率运用逆向衰减函数。计算方式如下。

.. code-block:: text

    if staircase == True:
         decayed_learning_rate = learning_rate / (1 + decay_rate * floor(global_step / decay_step))
    else:
         decayed_learning_rate = learning_rate / (1 + decay_rate * global_step / decay_step)

参数：
    - **learning_rate** (Variable|float)-初始学习率,类型可以为学习率变量(Variable)或float类型常量。
    - **decay_steps** (int) - 学习率衰减步长，见以上衰减运算。
    - **decay_rate** (float)- 学习率衰减率。见以上衰减运算。
    - **staircase** (bool) - 指定是否按阶梯状衰减。若为True, 学习率变化曲线呈阶梯状。若为False，学习率变化值曲线为平滑的曲线。默认值为False。
    - **begin** (int) - 起始步，默认值为0。
    - **step** (int) - 步大小，默认值为1。
    - **dtype**  (str) - 初始化学习率变量的dtype。默认值为"float32"。


**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    base_lr = 0.1
    with fluid.dygraph.guard():
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.InverseTimeDecay(
                  learning_rate=base_lr,
                  decay_steps=10000,
                  decay_rate=0.5,
                  staircase=True))



