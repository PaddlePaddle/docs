.. _cn_api_fluid_dygraph_CosineDecay:

CosineDecay
-------------------------------


.. py:class:: paddle.fluid.dygraph.CosineDecay(learning_rate, step_each_epoch, epochs, begin=0, step=1, dtype='float32')

:api_attr: 命令式编程模式（动态图)



该接口提供按余弦函数衰减学习率的功能。

余弦衰减的计算方式如下。

.. math::

    decayed\_learning\_rate = learning\_rate * 0.5 * (math.cos(global\_step * \frac{math.pi}{step\_each\_epoch} ) + 1)

式中，

- :math:`decayed\_learning\_rate` ： 衰减后的学习率。
式子中各参数详细介绍请看参数说明。

参数：
    - **learning_rate** (Variable | float) - 初始学习率。如果类型为Variable，则为shape为[1]的Tensor，数据类型为float32或float64；也可以是python的float类型。
    - **step_each_epoch** （int） - 遍历一遍训练数据所需的步数。
    - **begin** (int，可选) - 起始步，即以上公式中global_step的初始化值。默认值为0。
    - **step** (int，可选) - 步大小，即以上公式中global_step的每次的增量值。默认值为1。
    - **dtype**  (str，可选) - 初始化学习率变量的数据类型，可以为"float32", "float64"。默认值为"float32"。


**代码示例**

.. code-block:: python

    import paddle.fluid as fluid

    base_lr = 0.1
    with fluid.dygraph.guard():
        gru = fluid.dygraph.GRUUnit(5 * 3)
        optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.CosineDecay(
                base_lr, 10000, 120), parameter_list=gru.parameters())


