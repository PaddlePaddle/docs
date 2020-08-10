.. _cn_api_fluid_dygraph_ExponentialDecay:

ExponentialDecay
-------------------------------


.. py:class:: paddle.fluid.dygraph.ExponentialDecay(learning_rate, decay_steps, decay_rate, staircase=False, begin=0, step=1, dtype=’float32‘)

:api_attr: 命令式编程模式（动态图)



该接口提供一种学习率按指数函数衰减的功能。

指数衰减的计算方式如下。

.. math::

    decayed\_learning\_rate = learning\_rate * decay\_rate ^ y 


当staircase为False时，y对应的计算公式为：

.. math::

    y = \frac{global\_step}{decay\_steps} 

当staircase为True时，y对应的计算公式为：

.. math::

    y = math.floor(\frac{global\_step}{decay\_steps})

式中，

- :math:`decayed\_learning\_rate` ： 衰减后的学习率。
式子中各参数详细介绍请看参数说明。

参数：
    - **learning_rate** (Variable|float) - 初始学习率。如果类型为Variable，则为shape为[1]的Tensor，数据类型为float32或float64；也可以是python的float类型。
    - **decay_steps** (int) - 衰减步数。必须是正整数，该参数确定衰减周期。
    - **decay_rate** (float)- 衰减率。
    - **staircase** (bool) - 若为True，则以不连续的间隔衰减学习速率即阶梯型衰减。若为False，则以标准指数型衰减。默认值为False。
    - **begin** (int) - 起始步，即以上运算式子中global_step的初始化值。默认值为0。
    - **step** (int) - 步大小，即以上运算式子中global_step的每次的增量值，使得global_step随着训练的次数递增。默认值为1。
    - **dtype** (str) - 初始化学习率变量的数据类型，可以为"float32", "float64"。 默认值为"float32"。

返回： 无


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







