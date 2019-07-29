.. _cn_api_fluid_layers_asin:

asin
-------------------------------

.. py:function:: paddle.fluid.layers.asin(x, name=None)

arcsine激活函数。

.. math::
    out = sin^{-1}(x)

参数:
    - **x** - asin算子的输入

返回：        asin算子的输出。

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.asin(data)



