.. _cn_api_paddle_sinc:

sin
-------------------------------

.. py:function:: paddle.sinc(x, name=None)

计算输入的归一化 sinc 值。

计算公式为：

.. math::

    out_i =
    \left\{
    \begin{aligned}
    &1 & \text{ if $x_i = 0$} \\
    &\frac{\sin(\pi x_i)}{\pi x_i} & \text{ otherwise}
    \end{aligned}
    \right.

参数
::::::::::::

    - **x** (Tensor) - 输入 Tensor。数据类型为 bfloat16，float16，float32，float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
返回类型为 Tensor，数据类型同输入一致。

代码示例
::::::::::::

COPY-FROM: paddle.sinc
