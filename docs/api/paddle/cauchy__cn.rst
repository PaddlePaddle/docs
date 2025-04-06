.. _cn_api_paddle_cauchy_:

cauchy\_
------------------------------

.. py:function:: paddle.cauchy_(x: paddle.Tensor, loc: Numeric = 0, scale: Numeric = 1, name: str | None = None)

使用柯西分布(Cauchy distribution)随机数就地(Inplace)填充输入张量 x 。

参数
:::::::::
   -  **x** (Tensor) - 将被填充的张量，数据类型为 float32 或 float64。
   -  **loc** (scalar, 可选) - 分布峰值的位置。数据类型为 float32 或 float64。
   -  **scale** (scalar, 可选) - 半高宽(HWHM)。数据类型为 float32 或 float64。必须为正值。
   -  **name** (str|None, 可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

..

返回
:::::::::
输出 Tensor, 使用 Cauchy 分布中的数字填充的输入张量。


代码示例
:::::::::

COPY-FROM: paddle.cauchy_
