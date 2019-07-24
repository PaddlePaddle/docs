.. _cn_api_fluid_layers_atan:

atan
-------------------------------

.. py:function:: paddle.fluid.layers.atan(x, name=None)

arctanh激活函数。

.. math::
    out = tanh^{-1}(x)

参数:
    - **x** - atan算子的输入

返回：       atan算子的输出。

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.atan(data)





