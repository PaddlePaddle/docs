.. _cn_api_paddle_abs:

abs
-------------------------------

.. py:function:: paddle.abs(x, name=None)




绝对值函数。

.. math::
    out = |x|

参数
:::::::::
    - **x** (Tensor) - 输入的 Tensor，数据类型为：int32、int64、float16、float32、float64、complex64、complex128。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
输出 Tensor，与 ``x`` 维度相同、数据类型相同。

代码示例
:::::::::

COPY-FROM: paddle.abs
