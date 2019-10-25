.. _cn_api_fluid_layers_softplus:

softplus
-------------------------------

.. py:function:: paddle.fluid.layers.softplus(x,name=None)

softplus激活函数

.. math::
    out = \ln(1 + e^{x})

参数：
    - **x** (Variable) - 张量（Tensor）
    - **name** (str|None) - 该层名称（可选）。若设为None，则自动为该层命名。

返回: 张量(Tensor)

返回类型: 变量(Variable)

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(name="input", shape=[32, 784])
    result = fluid.layers.softplus(data)











