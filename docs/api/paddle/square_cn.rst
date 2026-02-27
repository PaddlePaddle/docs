.. _cn_api_paddle_square:

square
-------------------------------

.. py:function:: paddle.square(x, name=None, *, out=None)

对输入参数 x 进行逐元素取平方运算。

.. math::
    out = x^2

参数
::::::::::::
    - **x** (Tensor) - 任意维度的 Tensor，支持的数据类型：int32，int64，float32，float64，complex64，complex128。别名 ``input``。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
::::::::::::
    - **out** (Tensor，可选) - 输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 中，默认值为 ``None``。

返回
::::::::::::
取平方后的 Tensor，维度和数据类型同输入一致。

代码示例
::::::::::::

COPY-FROM: paddle.square
