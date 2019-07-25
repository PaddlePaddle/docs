.. _cn_api_fluid_layers_acos:

acos
-------------------------------

.. py:function:: paddle.fluid.layers.acos(x, name=None)

arccosine激活函数。

.. math::
    out = cos^{-1}(x)

参数:
    - **x** - acos算子的输入

返回：        acos算子的输出。

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.acos(data)


