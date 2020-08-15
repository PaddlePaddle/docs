.. _cn_api_fluid_dygraph_PolynomialDecay:

PolynomialDecay
-------------------------------


.. py:class:: paddle.fluid.dygraph.PolynomialDecay(learning_rate, decay_steps, end_learning_rate=0.0001, power=1.0, cycle=False, begin=0, step=1, dtype='float32')

:api_attr: 命令式编程模式（动态图)



该接口提供学习率按多项式衰减的功能。通过多项式衰减函数，使得学习率值逐步从初始的 ``learning_rate``，衰减到 ``end_learning_rate`` 。

计算方式如下。

若cycle为True，则计算公式为：

.. math::

    decay\_steps &= decay\_steps * math.ceil(\frac{global\_step}{decay\_steps})  \\
    decayed\_learning\_rate &= (learning\_rate-end\_learning\_rate)*(1-\frac{global\_step}{decay\_steps})^{power}+end\_learning\_rate

若cycle为False，则计算公式为：

.. math::

    global\_step &= min(global\_step, decay\_steps) \\
    decayed\_learning\_rate &= (learning\_rate-end\_learning\_rate)*(1-\frac{global\_step}{decay\_steps})^{power}+end\_learning\_rate

式中，

- :math:`decayed\_learning\_rate` ： 衰减后的学习率。
式子中各参数详细介绍请看参数说明。

参数：
    - **learning_rate** (Variable|float32) - 初始学习率。如果类型为Variable，则为shape为[1]的Tensor，数据类型为float32或float64；也可以是python的float类型。
    - **decay_steps** (int) - 衰减步数。必须是正整数，该参数确定衰减周期。
    - **end_learning_rate** (float，可选) - 最小的最终学习率。默认值为0.0001。
    - **power** (float，可选) - 多项式的幂。默认值为1.0。
    - **cycle** (bool，可选) - 学习率下降后是否重新上升。若为True，则学习率衰减到最低学习率值时，会出现上升。若为False，则学习率曲线则单调递减。默认值为False。
    - **begin** (int，可选) – 起始步，即以上运算式子中global_step的初始化值。默认值为0。
    - **step** (int，可选) – 步大小，即以上运算式子中global_step的递增值。默认值为1。
    - **dtype** (str，可选)– 初始化学习率变量的数据类型，可以为"float32", "float64"。默认值为"float32"。

返回： 无


**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    start_lr = 0.01
    total_step = 5000
    end_lr = 0
    with fluid.dygraph.guard():
        emb = fluid.dygraph.Embedding( [10, 10])
        optimizer  = fluid.optimizer.SGD(
            learning_rate = fluid.dygraph.PolynomialDecay(
            start_lr, total_step, end_lr, power=1.0),
            parameter_list = emb.parameters())
