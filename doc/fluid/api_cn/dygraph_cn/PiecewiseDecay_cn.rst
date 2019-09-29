.. _cn_api_fluid_dygraph_PiecewiseDecay:

PiecewiseDecay
-------------------------------

.. py:class:: paddle.fluid.dygraph.PiecewiseDecay(boundaries, values, begin, step=1, dtype='float32')

对初始学习率进行分段(piecewise)衰减。

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

参数：
    - **boundaries** -一列代表步数的数字
    - **values** -一列学习率的值，从不同的步边界中挑选
    - **begin**  – 用于初始化self.step_num的起始步(默认为0)。
    - **step**  – 计算新的step_num步号时使用的步大小(默认为1)。
    - **dtype**  – 初始化学习率变量的dtype


**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    boundaries = [10000, 20000]
    values = [1.0, 0.5, 0.1]
    with fluid.dygraph.guard():
        optimizer = fluid.optimizer.SGD(
           learning_rate=fluid.dygraph.PiecewiseDecay(boundaries, values, 0) )





