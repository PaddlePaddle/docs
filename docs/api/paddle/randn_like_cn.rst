.. _cn_api_paddle_randn_like:

randn_like
-------------------------------

.. py:function:: paddle.randn_like(x, dtype=None, name=None)

返回一个与输入张量尺寸相同的张量，其元素服从均值为 0、方差为 1 的标准正态分布。

参数
::::::::::
    - **x** (Tensor) – 输入的多维 Tensor，数据类型可以是 float16，bfloat16，float32，float64，complex64，complex128。输出 Tensor 的形状和 ``x`` 相同。如果 ``dtype`` 为 None，则输出 Tensor 的数据类型与 ``x`` 相同。
    - **dtype** (str|paddle.dtype|np.dtype，可选) - 输出 Tensor 的数据类型，支持 float16，bfloat16，float32，float64，complex64，complex128。当该参数值为 None 时，输出 Tensor 的数据类型与输入 Tensor 的数据类型一致。默认值为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    Tensor：服从均值为 0、方差为 1 的标准正态分布 Tensor，形状为 ``x.shape``，数据类型为 ``dtype``。

代码示例
:::::::::::

COPY-FROM: paddle.randn_like
