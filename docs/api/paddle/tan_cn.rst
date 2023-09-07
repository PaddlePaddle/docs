.. _cn_api_paddle_tan:

tan
-------------------------------

.. py:function:: paddle.tan(x, name=None)
三角函数 tangent。

输入范围是 `(k*pi-pi/2, k*pi+pi/2)`，输出范围是 `[-inf, inf]` 。

.. math::
    out = tan(x)

参数
:::::::::

  - **x** (Tensor) – 该 OP 的输入为 Tensor。数据类型为 float32，float64。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
:::::::::

Tensor - 该 OP 的输出为 Tensor，数据类型为输入一致。


代码示例
:::::::::

COPY-FROM: paddle.tan
