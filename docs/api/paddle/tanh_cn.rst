.. _cn_api_paddle_tanh:

tanh
-------------------------------

.. py:function:: paddle.tanh(x, name=None, *, out=None)


tanh 激活函数

.. math::
    out = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}

参数
:::::::::


    - **x** (Tensor) - Tanh 算子的输入，多维 Tensor，数据类型为 bfloat16、float16、float32、float64、uint8、int8、int16、int32 或 int64。别名 ``input``。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
:::::::::::
    - **out** (Tensor，可选) - 输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 中，默认值为 ``None``。

返回
:::::::::
tanh 的输出 Tensor，和输入具有相同的类型和 shape；整数类型输入会自动转换为 float32。


代码示例
:::::::::

COPY-FROM: paddle.tanh
