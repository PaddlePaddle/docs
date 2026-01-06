.. _cn_api_paddle_atan:

atan
-------------------------------

.. py:function:: paddle.atan(x, name=None, *, out=None)




arctangent 函数。

.. math::
    out = tan^{-1}(x)

参数
::::::::::::

    - **x** (Tensor) - 输入的 Tensor，数据类型为：float32、float64、float16、complex64、complex128。别名 ``input`` 。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
:::::::::
    - **out** (Tensor，可选) - 输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 中，默认值为 ``None``。

返回
::::::::::::
输出 Tensor，与 ``x`` 维度相同、数据类型相同。

代码示例
::::::::::::

COPY-FROM: paddle.atan
