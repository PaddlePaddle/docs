.. _cn_api_paddle_linalg_householder_product:

householder_product
-------------------------------

.. py:function:: paddle.linalg.householder_product(x, tau, name=None)


计算 Householder 矩阵乘积的前 n 列(输入矩阵为 `[*,m,n]` )。


该函数可以从矩阵 `x` (m x n) 得到向量 :math:`\omega_{i}`，其中前 `i-1` 个元素为零，第 i 个元素为 `1`，其余元素元素来自 `x` 的第 i 列。
并且使用向量 `tau` 可以计算 Householder 矩阵乘积的前 n 列。

.. math::
    H_i = I_m - \tau_i \omega_i \omega_i^H


参数
::::::::::::

    - **x** (Tensor): 形状为 `(*, m, n)` 的张量，其中 * 是零个或多个批量维度。
    - **tau** (Tensor): 形状为 `(*, k)` 的张量，其中 * 是零个或多个批量维度。
    - **name** (str, 可选): 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::

    - Tensor, dtype与输入张量相同, QR分解中的Q, :MATH:`out = q = H_1H_2H_3 ... H_K`

代码示例
::::::::::

COPY-FROM: paddle.linalg.householder_product
