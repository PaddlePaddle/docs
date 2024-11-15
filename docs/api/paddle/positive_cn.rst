.. _cn_api_paddle_positive:

positive
-------------------------------

.. py:function:: paddle.positive(x)

返回输入 Tensor 的本身。

计算公式为：

.. math::
    out=+x

参数
::::::::::::

    - **x** (Tensor) - 支持任意维度的 Tensor。数据类型为 uint8,int8,int16,int32,int64,float32，float64，float16，complex64，complex128。

返回
::::::::::::
返回类型为 Tensor,数据类型同输入一致。

代码示例
::::::::::::

COPY-FROM: paddle.positive
