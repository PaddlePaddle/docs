.. _cn_api_paddle_cauchy_:

cauchy\_
-------------------------------

.. py:function:: paddle.cauchy_(x, loc=0, scale=1, name=None)
该函数将输入 Tensor 填充为从柯西分布（Cauchy distribution）中采样的随机数。

参数
::::::::::::

    - **x** (Tensor) - 待填充的目标张量，数据类型必须为 float32 或 float64。
    - **loc** (scalar,可选) - 柯西分布的峰值位置，数据类型为 float32 或 float64，默认值为 0。
    - **scale** (scalar,可选) - 柯西分布的半高全宽（HWHM），必须为正数，数据类型为 float32 或 float64，默认值为 1。
    - **name** (str|None,可选) - 操作的名称，一般无需设置，默认值为 None。

返回
::::::::::::
填充了柯西分布随机数的输入张量

返回类型
::::::::::::
Tensor

代码示例
::::::::::::

COPY-FROM: paddle.cauchy_
