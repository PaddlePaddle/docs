.. _cn_api_fluid_layers_relu6:

relu6
-------------------------------

.. py:function:: paddle.fluid.layers.relu6(x, threshold=6.0, name=None)

relu6激活算子（Relu6 Activation Operator）

.. math::

    \\out=min(max(0, x), 6)\\


参数:
    - **x** (Variable) - Relu6 operator的输入
    - **threshold** (FLOAT|6.0) - Relu6的阈值
    - **name** (str|None) -这个层的名称(可选)。如果设置为None，该层将被自动命名。

返回: Relu6操作符的输出

返回类型: 输出(Variable)


**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name="x", shape=[3,10,32,32], dtype="float32")
    y = fluid.layers.relu6(x, threshold=6.0)







