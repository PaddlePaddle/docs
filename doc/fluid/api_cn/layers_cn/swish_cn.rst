.. _cn_api_fluid_layers_swish:

swish
-------------------------------

.. py:function:: paddle.fluid.layers.swish(x, beta=1.0, name=None)

Swish 激活函数

.. math::
         out = \frac{x}{1 + e^{- beta x}}

参数：
    - **x** (Variable) -  Swish operator 的输入
    - **beta** (浮点|1.0) - Swish operator 的常量beta
    - **name** (str|None) - 这个层的名称(可选)。如果设置为None，该层将被自动命名。

返回: Swish operator 的输出

返回类型: output(Variable)


**代码示例：**

.. code-block:: python

  import paddle.fluid as fluid
  x = fluid.layers.data(name="x", shape=[3,10,32,32], dtype="float32")
  y = fluid.layers.swish(x, beta=2.0)


