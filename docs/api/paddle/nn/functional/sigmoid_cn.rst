.. _cn_api_paddle_nn_functional_sigmoid:

sigmoid
-------------------------------

.. py:function:: paddle.nn.functional.sigmoid(x, name=None, *, out=None)



sigmoid 激活函数

.. math::
    sigmoid(x) = \frac{1}{1 + \mathrm{e}^{-x}}


参数
:::::::::

    - **x** (Tensor) - 激活函数的输入值，数据类型为 bfloat16、float16、float32、float64、uint8、int8、int16、int32、int64、complex64 或 complex128。别名 ``input``。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
:::::::::::
    - **out** (Tensor，可选) - 输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 中，默认值为 ``None``。

返回
:::::::::

Tensor，激活函数的输出值，形状与输入相同；整数类型输入会自动转换为 float32。

代码示例
:::::::::

COPY-FROM: paddle.nn.functional.sigmoid
