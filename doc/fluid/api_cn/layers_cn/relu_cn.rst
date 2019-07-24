.. _cn_api_fluid_layers_relu:

relu
-------------------------------

.. py:function:: paddle.fluid.layers.relu(x, name=None)

Relu接受一个输入数据(张量)，输出一个张量。将线性函数y = max(0, x)应用到张量中的每个元素上。

.. math::
              \\Out=\max(0,x)\\


参数:
  - **x** (Variable):输入张量。
  - **name** (str|None，默认None) :如果设置为None，该层将自动命名。

返回: 与输入形状相同的输出张量。

返回类型: 变量（Variable）

**代码示例**:

..  code-block:: python
      
    import paddle.fluid as fluid
    x = fluid.layers.data(name="x", shape=[3, 4], dtype="float32")
    output = fluid.layers.relu(x)










