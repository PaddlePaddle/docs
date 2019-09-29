.. _cn_api_fluid_dygraph_PiecewiseDecay:

PiecewiseDecay
-------------------------------

.. py:class:: paddle.fluid.dygraph.PiecewiseDecay(boundaries, values, begin, step=1, dtype='float32')

该接口提供对初始学习率进行分段(piecewise)常数衰减的功能。需要事先定义好的训练步数区间，在对应区间置不同的学习率的常数值。

分段常数衰减的过程举例描述如下。

.. code-block:: text

    例如，设定的boundaries列表为[10000, 20000]，候选学习率常量列表values为[1.0, 0.5, 0.1]，则：
    1、在训练步数step小于10000步，学习率值为1.0。
    2、在训练步数step大于等于10000步，并且小于20000步时，学习率值为0.5。
    3、在训练步数step大于20000步时，学习率值为0.1。

参数：
    - **boundaries** (list) - 指定衰减的步数边界。列表的数据元素为Python int类型。
    - **values** (list) - 备选学习率列表。数据元素类型为Python float的列表。与边界值列表有对应的关系。
    - **begin** (int) – 起始步，即以上举例描述中step的初始化值。
    - **step** (int，可选) – 步大小，即以上举例描述中step的递增值。默认值为1。
    - **dtype** (str，可选) – 初始化学习率变量的数据类型。默认值为"float32"。


**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    boundaries = [10000, 20000]
    values = [1.0, 0.5, 0.1]
    with fluid.dygraph.guard():
        optimizer = fluid.optimizer.SGD(
           learning_rate=fluid.dygraph.PiecewiseDecay(boundaries, values, 0) )





