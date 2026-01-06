.. _cn_api_paddle_asinh:

asinh
-------------------------------

.. py:function:: paddle.asinh(x, name=None, *, out=None)

Arcsinh 函数。

.. math::
    out = asinh(x)

参数
:::::::::
    - **x** (Tensor) - 输入的 Tensor，数据类型为：float32、float64、complex64、complex128。别名 ``input``。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
:::::::::
    - **out** (Tensor，可选) - 输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 中，默认值为 ``None``。

返回
:::::::::
输出 Tensor，与 ``x`` 维度相同、数据类型相同。



代码示例
:::::::::

COPY-FROM: paddle.asinh
