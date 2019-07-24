.. _cn_api_fluid_layers_soft_relu:

soft_relu
-------------------------------

.. py:function:: paddle.fluid.layers.soft_relu(x, threshold=40.0, name=None)

SoftRelu 激活函数

.. math::   out=ln(1+exp(max(min(x,threshold),threshold))

参数:
    - **x** (variable) - SoftRelu operator的输入
    - **threshold** (FLOAT|40.0) - SoftRelu的阈值
    - **name** (str|None) - 该层的名称(可选)。如果设置为None，该层将被自动命名

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid

    x = fluid.layers.data(name=”x”, shape=[2,3,16,16], dtype=”float32”)
    y = fluid.layers.soft_relu(x, threshold=20.0)








