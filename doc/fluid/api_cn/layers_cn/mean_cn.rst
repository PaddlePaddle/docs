.. _cn_api_fluid_layers_mean:

mean
-------------------------------

.. py:function:: paddle.fluid.layers.mean(x, name=None)

mean算子计算X中所有元素的平均值

参数：
        - **x** (Variable)- (Tensor) 均值运算的输入。
        - **name** (basestring | None)- 输出的名称。

返回：       均值运算输出张量（Tensor）

返回类型：        Variable

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    input = fluid.layers.data(
        name='data', shape=[2, 3], dtype='float32')
    mean = fluid.layers.mean(input)









