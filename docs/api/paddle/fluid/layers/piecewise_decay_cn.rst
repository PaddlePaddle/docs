.. _cn_api_fluid_layers_piecewise_decay:

piecewise_decay
-------------------------------

.. py:function:: paddle.fluid.layers.piecewise_decay(boundaries,values)




对初始学习率进行分段衰减。

该算法可用如下代码描述。

.. code-block:: text

    boundaries = [10000, 20000]
    values = [1.0, 0.5, 0.1]
    if step < 10000:
        learning_rate = 1.0
    elif 10000 <= step < 20000:
        learning_rate = 0.5
    else:
        learning_rate = 0.1

参数
::::::::::::

    - **boundaries(list)** - 代表步数的数字
    - **values(list)** - 学习率的值，不同的步边界中的学习率值

返回
::::::::::::
衰减的学习率

代码示例
::::::::::::

.. code-block:: python

        import paddle.fluid as fluid
        boundaries = [10000, 20000]
        values = [1.0, 0.5, 0.1]
        optimizer = fluid.optimizer.Momentum(
            momentum=0.9,
            learning_rate=fluid.layers.piecewise_decay(boundaries=boundaries, values=values),
            regularization=fluid.regularizer.L2Decay(1e-4))






