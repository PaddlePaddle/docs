.. _cn_api_fluid_layers_isfinite:

isfinite
-------------------------------

.. py:function:: paddle.fluid.layers.isfinite(x)

测试x是否包含无穷大/NAN值，如果所有元素都是有穷数，返回Ture,否则返回False

参数：
  - **x(variable)** - 用于被检查的Tensor/LoDTensor

返回: Variable: tensor变量存储输出值，包含一个bool型数值

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    var = fluid.layers.data(name="data",
                            shape=(4, 6),
                            dtype="float32")
    out = fluid.layers.isfinite(v)



