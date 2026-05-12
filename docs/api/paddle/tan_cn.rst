.. _cn_api_paddle_tan:

tan
-------------------------------

.. py:function:: paddle.tan(x, name=None, *, out=None)
三角函数 tangent。

输入范围是 ``(k*pi-pi/2, k*pi+pi/2)``，输出范围是 ``[-inf, inf]`` 。

.. math::
    out = tan(x)

参数
:::::::::
    - **x** (Tensor) - 该 OP 的输入为 Tensor。数据类型为 float32、float64、float16、bfloat16、uint8、int8、int16、int32、int64、complex64 或 complex128。别名 ``input``。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
:::::::::
    - **out** (Tensor，可选) - 输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 中，默认值为 ``None``。

返回
:::::::::
Tensor - 该 OP 的输出为 Tensor，数据类型与输入一致（整数类型会自动转换为 float32）。


代码示例
:::::::::

COPY-FROM: paddle.tan
