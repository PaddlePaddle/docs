.. _cn_api_paddle_digamma:

digamma
----------------

.. py:function:: paddle.digamma(x, name=None)


逐元素计算输入Tensor的digamma函数值

.. math::
    \\Out = \Psi(x) = \frac{ \Gamma^{'}(x) }{ \Gamma(x) }\\


参数
:::::::::
  - **x** (Tensor) – 输入Tensor。数据类型为float32，float64。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
``Tensor``, digamma函数计算结果，数据类型和维度大小与输入一致。

代码示例
:::::::::

COPY-FROM: paddle.digamma
