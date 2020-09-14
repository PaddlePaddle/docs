.. _cn_api_fluid_dygraph_NaturalExpDecay:

NaturalExpDecay
-------------------------------


.. py:class:: paddle.fluid.dygraph.NaturalExpDecay(learning_rate, decay_steps, decay_rate, staircase=False, begin=0, step=1, dtype='float32')




该接口提供按自然指数衰减学习率的功能。

自然指数衰减的计算方式如下。

.. math::

    decayed\_learning\_rate = learning\_rate * e^{y} 

当staircase为False时，y对应的计算公式为：

.. math::

    y = - decay\_rate * \frac{global\_step}{decay\_steps}

当staircase为True时，y对应的计算公式为：

.. math::

    y = - decay\_rate * math.floor(\frac{global\_step}{decay\_steps}) 

式中，

- :math:`decayed\_learning\_rate` ： 衰减后的学习率。
式子中各参数详细介绍请看参数说明。

参数：
    - **learning_rate** (Variable|float) - 初始学习率值。如果类型为Variable，则为shape为[1]的Tensor，数据类型为float32或float64；也可以是python的float类型。
    - **decay_steps** (int) – 指定衰减的步长。该参数确定衰减的周期。
    - **decay_rate** (float) – 指定衰减率。
    - **staircase** (bool，可选) - 若为True, 学习率变化曲线呈阶梯状，若为False，学习率变化值曲线为平滑的曲线。默认值为False。
    - **begin** (int，可选) – 起始步，即以上运算式子中global_step的初始化值。默认值为0。
    - **step** (int，可选) – 步大小，即以上运算式子中global_step的每次的增量值。默认值为1。
    - **dtype** (str，可选) – 学习率值的数据类型，可以为"float32", "float64"。默认值为"float32"。

返回： 无

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    base_lr = 0.1
    with fluid.dygraph.guard():
        emb = fluid.dygraph.Embedding([10, 10])
        sgd_optimizer = fluid.optimizer.SGD(
                learning_rate=fluid.dygraph.NaturalExpDecay(
                    learning_rate=base_lr,
                    decay_steps=10000,
                    decay_rate=0.5,
                    staircase=True),
                parameter_list=emb.parameters())





