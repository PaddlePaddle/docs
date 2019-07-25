.. _cn_api_fluid_layers_has_nan:

has_nan
-------------------------------

.. py:function:: paddle.fluid.layers.has_nan(x)

测试x是否包含NAN

参数：
  - **x(variable)** - 用于被检查的Tensor/LoDTensor

返回： tensor变量存储输出值，包含一个bool型数值

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(name="input", shape=[4, 32, 32], dtype="float32")
    res = fluid.layers.has_nan(data)




