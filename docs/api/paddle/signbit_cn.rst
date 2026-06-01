.. _cn_api_paddle_signbit:

signbit
-------------------------------

.. py:function:: paddle.signbit(x, name=None, *, out=None)

对输入参数 ``x`` 的每个元素判断是否设置了其符号位，并输出判断值。若存在符号位，则输出 True，否则输出 False。

参数
::::::::::::
    - **x** (Tensor) – 进行符号位判断的多维 Tensor，数据类型为 uint8，int8，int16，int32，int64，bfloat16，float16，float32 或 float64。别名 ``input``。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
::::::::::::
    - **out** (Tensor，可选) - 输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 中，默认值为 ``None``。

返回
::::::::::::
Tensor，输出掩码，数据的 shape 大小及数据类型和输入 ``x`` 一致。


代码示例
::::::::::::

COPY-FROM: paddle.signbit
