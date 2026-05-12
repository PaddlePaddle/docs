.. _cn_api_paddle_inverse:

inverse
-------------------------------

.. py:function:: paddle.inverse(x, name=None, *, out=None)

计算方阵的逆矩阵。方阵是行数和列数相同的矩阵。输入可以是一个方阵（2-D Tensor）或者多个方阵组成的 batch（batch of square matrices）。

参数
::::::::::::
    - **x** (Tensor) - 输入 Tensor，最后两个维度应相等。当维度大于 2 时，视为方阵的 batch。数据类型为 float32、float64、complex64、complex128。别名 ``input``。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
::::::::::::
    - **out** (Tensor，可选) - 输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 中，默认值为 ``None``。

返回
::::::::::::
Tensor，``x`` 的逆矩阵，维度和数据类型与 ``x`` 相同。

代码示例
::::::::::::

COPY-FROM: paddle.inverse
