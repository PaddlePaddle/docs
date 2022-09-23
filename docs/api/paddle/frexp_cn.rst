.. _cn_api_paddle_frexp:

frexp
-------------------------------

.. py:function:: paddle.frexp(x, name)


用于把一个浮点数分解为尾数和指数的函数, 返回一个尾数 Tensor 和一个指数 Tensor

参数
::::::::::
    - **x** (Tensor) – 输入是一个多维的 Tensor，它的数据类型可以是 float32，float64。
    - **name** (str，可选) - 具体用法请参见  :ref:`api_guide_Name` ，一般无需设置，默认值为 None。
返回
::::::::::
    mantissa（Tensor）：分解后的尾数，类型为 Tensor，形状和原输入的形状一致。

    exponent（Tensor）：分解后的指数，类型为 Tensor，形状和原输入的形状一致。


代码示例
::::::::::

COPY-FROM: paddle.frexp
