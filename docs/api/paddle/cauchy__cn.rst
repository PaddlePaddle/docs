.. _cn_api_paddle_cauchy_:

cauchy\_
------------------------------

.. py:function::   paddle.cauchy_(x: paddle.Tensor, loc: Numeric = 0, scale: Numeric =
   1, name: str \| None = None) →
   paddle.Tensor[`source <https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/creation.py#L3215>`__]

使用 Cauchy 分布中的数字填充张量。

参数
:::::::::
   -  **x** (Tensor) – 将被填充的张量，数据类型为 float32 或 float64。
   -  **loc** (scalar, optional) – 分布峰值的位置。数据类型为 float32 或
      float64。
   -  **scale** (scalar, optional) – 半高宽（HWHM）。数据类型为 float32 或
      float64。必须为正值。
   -  **name** (str|None, optional) –
      具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

..

返回
:::::::::
使用 Cauchy 分布中的数字填充的输入张量。

返回类型
:::::::::
Tensor

代码示例
:::::::::

COPY-FROM: paddle.cauchy_
