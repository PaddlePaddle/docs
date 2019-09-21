.. _cn_api_fluid_layers_has_nan:

has_nan
-------------------------------

.. py:function:: paddle.fluid.layers.has_nan(x)

检查输入的变量(x)中是否包含NAN。

参数：
  - **x** (Variable) - 被检查的变量tensor/LoDtensor。

返回：Variable(tensor)变量存储输出值，包含一个bool型数值，指明输入中是否包含NAN。

返回类型：变量(Variable)

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(name="input", shape=[4, 32, 32], dtype="float32")
    res = fluid.layers.has_nan(data)




