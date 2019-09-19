.. _cn_api_fluid_layers_pow:

pow
-------------------------------

.. py:function:: paddle.fluid.layers.pow(x, factor=1.0, name=None)

指数激活算子（Pow Activation Operator.）

**注意**：如果使用`elementwise_pow`，请查看相关文档 :ref:`_cn_api_fluid_layers_elementwise_pow` 。

.. math::

    out = x^{factor}

参数
    - **x** (Variable) - Pow operator的输入。
    - **factor** (FLOAT|Variable|1.0) - Pow的指数因子。
    - **name** (str|None) -这个层的名称(可选)。如果设置为None，该层将被自动命名。

返回: 输出Pow操作符

返回类型: 输出(Variable)


**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid

    x = fluid.layers.data(name="x", shape=[3,10,32,32], dtype="float32")

    # example 1: argument factor is float
    y_1 = fluid.layers.pow(x, factor=2.0)
    # y_1 is x^{2.0}

    # example 2: argument factor is Variable
    factor_tensor = fluid.layers.fill_constant([1], "float32", 3.0)
    y_2 = fluid.layers.pow(x, factor=factor_tensor)
    # y_2 is x^{2.0}






