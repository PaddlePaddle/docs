.. _cn_api_fluid_layers_sums:

sums
-------------------------------

.. py:function:: paddle.fluid.layers.sums(input,out=None)

计算多个输入Tensor的和。

参数：
    - **input** (list) - 多个维度相同的Tensor组成的元组。支持的数据类型：float32，float64，int32，int64。
    - **out** (Variable，可选) - 求和的结果。默认值为None。

返回：输入的和。若 ``out`` 不为 ``None`` ，返回 ``out`` 。

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid

    # 多个Tensor求和
    x0 = fluid.layers.fill_constant(shape=[16, 32], dtype='int64', value=1)
    x1 = fluid.layers.fill_constant(shape=[16, 32], dtype='int64', value=2)
    x2 = fluid.layers.fill_constant(shape=[16, 32], dtype='int64', value=3)
    sums = fluid.layers.sums(input=[x0, x1, x2])
