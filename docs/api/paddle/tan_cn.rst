.. _cn_api_fluid_layers_tan:

tan
-------------------------------

.. py:function:: paddle.tan(x, name=None)
三角函数tangent。

输入范围是 `(k*pi-pi/2, k*pi+pi/2)`，输出范围是 `[-inf, inf]` 。

.. math::
    out = tan(x)

参数
:::::::::

  - **x** (Tensor) – 该OP的输入为Tensor。数据类型为float32，float64。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
:::::::::

Tensor - 该OP的输出为Tensor，数据类型为输入一致。


代码示例
:::::::::

COPY-FROM: paddle.tan