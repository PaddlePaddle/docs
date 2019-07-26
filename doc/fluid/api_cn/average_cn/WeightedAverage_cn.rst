.. _cn_api_fluid_average_WeightedAverage:

WeightedAverage
-------------------------------

.. py:class:: paddle.fluid.average.WeightedAverage

计算加权平均值。

平均计算完全通过Python完成。它们不会改变Paddle的程序，也不会修改NN模型的配置。它们完全是Python函数的包装器。

**示例代码**

.. code-block:: python

            import paddle.fluid as fluid
            avg = fluid.average.WeightedAverage()
            avg.add(value=2.0, weight=1)
            avg.add(value=4.0, weight=2)
            avg.eval()
            # 结果为 3.333333333.
            # 因为 (2.0 * 1 + 4.0 * 2) / (1 + 2) = 3.333333333











































