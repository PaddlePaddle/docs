.. _cn_api_paddle_frexp:

frexp
-------------------------------

.. py:function:: paddle.frexp(x, name=None, *, out=None)

用于把一个浮点数分解为尾数和指数的函数，返回一个尾数 Tensor 和一个指数 Tensor。

参数
::::::::::
    - **x** (Tensor) – 输入是一个多维的 Tensor，它的数据类型可以是 float32，float64。别名 ``input``。
    - **name** (str，可选) - 具体用法请参见  :ref:`api_guide_Name` ，一般无需设置，默认值为 None。

关键字参数
::::::::::
    - **out** (tuple[Tensor, Tensor]，可选) - 输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 元组中，默认值为 ``None``。

返回
::::::::::
    mantissa（Tensor）：分解后的尾数，类型为 Tensor，形状和原输入的形状一致。

    exponent（Tensor）：分解后的指数，类型为 Tensor，形状和原输入的形状一致。


代码示例
::::::::::

COPY-FROM: paddle.frexp
