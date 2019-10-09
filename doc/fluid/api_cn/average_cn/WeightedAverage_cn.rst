.. _cn_api_fluid_average_WeightedAverage:

WeightedAverage
-------------------------------

.. py:class:: paddle.fluid.average.WeightedAverage

计算加权平均值。只能计算Python的数值和numpy的ndarray，内部由累加器来存储相应的值。

该类包含以下方法

reset()

重置计数器

返回： 无


add(value, weight)

往累加器添加值和weight信息

参数：
    - **value (int|float|ndarray)**  - 往累加器中添加的value值
    - **weight (int|float|ndarray)**  – value对应的权重。如果类型为ndarray，shape必须等于[1]

返回：无

eval()

根据添加的值和weight信息，计算加权平均值

如果没有添加任何值，会报出ValueError的异常

返回：所有的添加到计数器中所有value的值

返回类型：数据类型和value一致，如果添加的value同时包含python（int或float）和和numpy的ndarray，则返回numpy的ndarray

**示例代码**

.. code-block:: python

            import paddle.fluid as fluid
            avg = fluid.average.WeightedAverage()
            avg.add(value=2.0, weight=1)
            avg.add(value=4.0, weight=2)
            avg.eval()
            # 结果为 3.333333333.
            # 因为 (2.0 * 1 + 4.0 * 2) / (1 + 2) = 3.333333333











































