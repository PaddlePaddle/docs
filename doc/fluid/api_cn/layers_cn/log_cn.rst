.. _cn_api_fluid_layers_log:

log
-------------------------------

.. py:function:: paddle.fluid.layers.log(x, name=None)


给定输入张量，计算其每个元素的自然对数

.. math::
                  \\Out=ln(x)\\


参数:
  - **x** (Variable) – 输入张量
  - **name** (str|None, default None) – 该layer的名称，如果为None，自动命名

返回：给定输入张量计算自然对数

返回类型: 变量（variable）


**代码示例**

..  code-block:: python

  import paddle.fluid as fluid
  x = fluid.layers.data(name="x", shape=[3, 4], dtype="float32")
  output = fluid.layers.log(x)











