.. _cn_api_paddle_log1p:

log1p
-------------------------------

.. py:function:: paddle.log1p(x, name=None, *, out=None)


计算 Log1p（自然对数 + 1）结果。

.. math::
                  \\Out=ln(x+1)\\


参数
::::::::::::

  - **x** (Tensor) – 输入为一个多维的 Tensor，数据类型为 int32，int64，float16，bfloat16，float32， float64， complex64 或 complex128。别名 ``input``。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
:::::::::::
    - **out** (Tensor，可选) - 输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 中，默认值为 ``None``。

返回
::::::::::::
计算 ``x`` 的自然对数 + 1 后的 Tensor，数据类型，形状与 ``x`` 一致。

代码示例
::::::::::::

COPY-FROM: paddle.log1p
