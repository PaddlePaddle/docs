.. _cn_api_fluid_layers_sum:

sum
-------------------------------

.. py:function:: paddle.fluid.layers.sum(x)

sum算子。

该算子对输入张量求和。所有输入都可以携带LoD（详细程度）信息，但是输出仅与第一个输入共享LoD信息。

参数：
        - **x** （Variable）- （vector <Tensor>）sum算子的输入张量（Tensor）。

返回:        (Tensor）求和算子的输出张量。

返回类型：        Variable


**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.layers as layers
    input0 = fluid.layers.data(name="input0", shape=[13, 11], dtype='float32')
    input1 = layers.data(name="input1", shape=[13, 11], dtype='float32')
    out = fluid.layers.sum([input0,input1])







