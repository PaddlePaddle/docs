.. _cn_api_paddle_sin:

sin
-------------------------------

.. py:function:: paddle.sin(x, name=None, *, out=out)

计算输入的正弦值。

计算公式为：

.. math::
    out=sin(x)

参数
::::::::::::

    - **x** (Tensor) - 支持任意维度的 Tensor。数据类型为 float32，float64，float16，complex64，complex128。别名  ``input`` 。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
:::::::::

    - **out** (Tensor，可选) - 输出 Tensor，若不为  ``None`` ，计算结果将保存在该 Tensor 中，默认值为  ``None`` 。

返回
::::::::::::
返回类型为 Tensor，数据类型同输入一致。

代码示例
::::::::::::

COPY-FROM: paddle.sin
