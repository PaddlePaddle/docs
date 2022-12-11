.. _cn_api_linalg_matrix_power:

matrix_power
-------------------------------

.. py:function:: paddle.linalg.matrix_power(x, n, name=None)


计算一个或一批方阵的 ``n`` 次幂。

记 :math:`X` 为一个或一批方阵，:math:`n` 为幂次，则公式为：

.. math::
    Out = X ^ {n}

特别地，

- 如果 ``n > 0``，则返回计算 ``n`` 次幂后的一个或一批矩阵。

- 如果 ``n = 0``，则返回一个或一批单位矩阵。

- 如果 ``n < 0``，则返回每个矩阵的逆（若矩阵可逆）的 ``abs(n)`` 次幂。

参数
:::::::::
    - **x** (Tensor)：输入的欲进行 ``n`` 次幂运算的一个或一批方阵，类型为 Tensor。 ``x`` 的形状应为 ``[*, M, M]``，其中 ``*`` 为零或更大的批次维度，数据类型为 float32， float64。
    - **n** (int)：输入的幂次，类型为 int。它可以是任意整数。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

Tensor，这个（或这批）矩阵 ``x`` 经过 ``n`` 次幂运算后的结果，数据类型和输入 ``x`` 的一致。

代码示例
::::::::::

COPY-FROM: paddle.linalg.matrix_power
