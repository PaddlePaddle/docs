.. _cn_api_fluid_layers_logsigmoid:

logsigmoid
-------------------------------

.. py:function:: paddle.fluid.layers.logsigmoid(x, name=None)

:alias_main: paddle.nn.functional.logsigmoid
:alias: paddle.nn.functional.logsigmoid,paddle.nn.functional.activation.logsigmoid
:old_api: paddle.fluid.layers.logsigmoid



Logsigmoid激活函数


.. math::

    out = \log \frac{1}{1 + e^{-x}}


参数:
    - **x** (Variable)- 张量（Tensor）
    - **name** (str|None) - 该层名称（可选），若设为None，则自动为该层命名。

返回: 张量(Tensor)

返回类型: 变量(Variable)

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(name="input", shape=[32, 784])
    result = fluid.layers.logsigmoid(data)









