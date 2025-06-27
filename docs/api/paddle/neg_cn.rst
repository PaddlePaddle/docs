.. _cn_api_paddle_neg:

neg
-------------------------------

.. py:function:: paddle.neg(x, name=None)




计算输入 x 的相反数并返回。

.. math::
    out = -x

参数
:::::::::
    - **x** (Tensor) - 输入的 Tensor，数据类型为：bfloat16、float16、float32、float64、int8、int16、int32、int64、uint8、complex64、complex128。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
输出 Tensor，与 ``x`` 维度相同、数据类型相同。

代码示例
:::::::::

COPY-FROM: paddle.neg
