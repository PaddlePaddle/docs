.. _cn_api_fluid_layers_isfinite:

isfinite
-------------------------------

.. py:function:: paddle.fluid.layers.isfinite(x)




``注意：此算子的输入 Tensor / LoDTensor 数据类型必须为 int32 / float / double 之一。``

测试 x 是否包含无穷值（即 nan 或 inf）。若元素均为有穷数，返回真；否则返回假。

参数
::::::::::::

  - **x(variable)**：变量，包含被测试的 Tensor / LoDTensor。

返回
::::::::::::
 
  - Variable (Tensor / LoDTensor)，此 Tensor 变量包含一个 bool 型结果。

返回类型
  - Variable (Tensor / LoDTensor)，一个包含 Tensor 的变量。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.isfinite