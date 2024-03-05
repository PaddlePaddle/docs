.. _cn_api_paddle_conj:

conj
-------------------------------

.. py:function:: paddle.conj(x, name=None)


是逐元素计算 Tensor 的共轭运算。

参数
::::::::::::

    - **x** (Tensor) - 输入的复数值的 Tensor，数据类型为：complex64、complex128、float16、float32、float64、int32 或 int64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

Tensor，输入的共轭。形状和数据类型与输入一致。如果 Tensor 元素是实数类型，如 float32、float64、int32、或者 int64，返回值和输入相同。


代码示例
::::::::::::

COPY-FROM: paddle.conj
