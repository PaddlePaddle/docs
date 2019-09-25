.. _cn_api_fluid_layers_sum:

sum
-------------------------------

.. py:function:: paddle.fluid.layers.sum(x)

该OP用于对输入的一至多个Tensor或LoDTensor求和。如果输入的是LoDTensor，输出仅与第一个输入共享LoD信息（序列信息）。

参数：
    **x** (Variable|list(Variable)) - 输入的一至多个Variable。Variable可以是Tensor或LoDTensor，数据类型支持：float32，float64，int8，int32，int64。如果输入了多个Variable，则不同Variable间的形状和数据类型应保持一致。

返回：对输入 ``x`` 中的Variable求和后的Tensor或LoDTensor，其形状和数据类型与 ``x`` 一致

返回类型：Variable


**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.layers as layers
    input0 = layers.data(name="input0", shape=[13, 11], dtype='float32')
    input1 = layers.data(name="input1", shape=[13, 11], dtype='float32')
    out = layers.sum([input0,input1])







