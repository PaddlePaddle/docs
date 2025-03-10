.. _cn_api_paddle_linalg_lu_solve:

lu_solve
-------------------------------

.. py:function:: paddle.linalg.lu_solve(b, lu, pivots, trans="N", name=None)

给定 `A` 的 LU 分解结果 和列向量 `b` ，求解线性方程组的解 `x`。

记 :math:`A` 为一个或一批方阵，:math:`b` 一个或一批矩阵，当 `trans` 为 `N` 时，公式为：

.. math::
    b = A * X

当 `trans` 为 `T` 时，公式为：

.. math::
    b = A ^ {T} * X

当 `trans` 为 `C` 时，公式为：

.. math::
    b = A ^ {H} * X

.. note::

    `lu` 和 `pivots` 由 ``paddle.linalg.lu`` 得到。

参数
::::::::::::

    - **b** (Tensor) - 输入的欲进行线性方程组求解的右值，类型为 Tensor。 ``b`` 的形状应为 ``[*, M, K]``，其中 ``*`` 为零或更大的批次维度，数据类型为 float32， float64。
    - **lu** (Tensor) - LU 分解结果矩阵，由 L、U 拼接组成，类型为 Tensor。 ``lu`` 的形状应为 ``[*, M, M]``，其中 ``*`` 为零或更大的批次维度。数据类型和 ``b`` 相同。
    - **pivots** (Tensor) - LU 分解结果的主元信息，类型为 Tensor。 ``pivots`` 的形状应为 ``[*, M]``，其中 ``*`` 为零或更大的批次维度。数据类型为 int32。
    - **trans** (str，可选) - 是否对 A 进行转置，该参数的合法值为 'N'，'T'，'C'，默认值为 N。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

    - Tensor，这个（或这批）矩阵 ``lu`` 、 ``pivots`` 和 ``b`` 经过运算后的结果，数据类型及维度和输入 ``b`` 的一致。

代码示例
::::::::::

COPY-FROM: paddle.linalg.lu_solve
