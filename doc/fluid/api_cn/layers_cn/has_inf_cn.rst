.. _cn_api_fluid_layers_has_inf:

has_inf
-------------------------------

.. py:function:: paddle.fluid.layers.has_inf(x)

检查输入的变量(x)中是否包含无穷数(inf)。

参数：
    - **x** (Variable) - 被检查的变量Tensor/LoDTensor。

返回：Variable(Tensor)变量存储输出值，包含一个bool型数值，指明输入中是否包含无穷数(inf)。

返回类型：变量(Variable)

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(name="input", shape=[4, 32, 32], dtype="float32")
    res = fluid.layers.has_inf(data)









