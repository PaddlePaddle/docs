.. _cn_api_fluid_layers_sum:

sum
-------------------------------

.. py:function:: paddle.fluid.layers.sum(x)

该OP用于对输入的多个Tensor或LoDTensor求和。如果输入的是LoDTensor，输出仅与第一个输入共享LoD信息（序列信息）。

参数：
    **x** (list[Variable]) - 输入的多个Variable，用list[Variable]表示。Variable可以是Tensor或LoDTensor，不同Variable之间的形状与类型必须相同。

返回：对输入 ``x`` 中各个Variable求和后的结果，与输入 ``x`` 的形状、类型相同

返回类型：Variable


**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.layers as layers
    input0 = layers.data(name="input0", shape=[13, 11], dtype='float32')
    input1 = layers.data(name="input1", shape=[13, 11], dtype='float32')
    out = layers.sum([input0,input1])







