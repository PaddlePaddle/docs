.. _cn_api_paddle_tensor_math_frexp:

frexp
-------------------------------

.. py:function:: paddle.frexp(x)


用于把一个浮点数分解为尾数和指数的函数, 返回一个尾数 Tensor 和一个指数 Tensor

参数
::::::::::
    - **x** (Tensor) – 输入是一个多维的 Tensor，它的数据类型可以是 float32，float64。

返回
::::::::::
    mantissa（Tensor）：分解后的尾数，类型为 Tensor，形状和原输入的形状一致。
    exponent（Tensor）：分解后的指数，类型为 Tensor，形状和原输入的形状一致。


代码示例
::::::::::

COPY-FROM: paddle.tensor.math.frexp
