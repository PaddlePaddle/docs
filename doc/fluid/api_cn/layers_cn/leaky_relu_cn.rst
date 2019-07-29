.. _cn_api_fluid_layers_leaky_relu:

leaky_relu
-------------------------------

.. py:function:: paddle.fluid.layers.leaky_relu(x, alpha=0.02, name=None)

LeakyRelu 激活函数

.. math::   out=max(x,α∗x)

参数:
    - **x** (Variable) - LeakyRelu Operator的输入
    - **alpha** (FLOAT|0.02) - 负斜率，值很小。
    - **name** (str|None) - 此层的名称(可选)。如果设置为None，该层将被自动命名。

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name="x", shape=[2,3,16,16], dtype="float32")
    y = fluid.layers.leaky_relu(x, alpha=0.01)







