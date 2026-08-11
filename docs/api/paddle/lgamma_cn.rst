.. _cn_api_paddle_lgamma:

lgamma
-------------------------------

.. py:function:: paddle.lgamma(x, name=None, *, out=None)

计算输入 x 的 gamma 函数的自然对数并返回。


参数
:::::::::
    - **x** (Tensor) - 输入 Tensor，数据类型为 bfloat16、float16、float32、float64、uint8、int8、int16、int32 或 int64。别名 ``input``。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
:::::::::::
    - **out** (Tensor，可选) - 输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 中，默认值为 ``None``。

返回
:::::::::
输出 Tensor，与 ``x`` 维度相同、数据类型相同；整数类型输入会自动转换为 float32。

代码示例
:::::::::

COPY-FROM: paddle.lgamma
