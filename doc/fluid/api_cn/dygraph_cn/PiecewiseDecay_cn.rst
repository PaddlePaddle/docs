.. _cn_api_fluid_dygraph_PiecewiseDecay:

PiecewiseDecay
-------------------------------

.. py:class:: paddle.fluid.dygraph.PiecewiseDecay(boundaries, values, begin, step=1, dtype='float32')

该接口提供对初始学习率进行分段(piecewise)衰减的功能。

分段衰减的计算方式如下。

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
    - **boundaries** (list) - 指定衰减的步数边界。列表的数据元素为int类型。
    - **values** (list) - 备选学习率列表。数据元素类型为float的列表。与边界值列表有一一对应的关系，例如在计算方式示例中，小于1000步的学习率均为1.0。
    - **begin**  – 起始步。默认值为0。
    - **step**  – 步大小。默认值为1。
    - **dtype**  – 初始化学习率变量的dtype。默认值为"float32"。


**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    boundaries = [10000, 20000]
    values = [1.0, 0.5, 0.1]
    with fluid.dygraph.guard():
        optimizer = fluid.optimizer.SGD(
           learning_rate=fluid.dygraph.PiecewiseDecay(boundaries, values, 0) )





