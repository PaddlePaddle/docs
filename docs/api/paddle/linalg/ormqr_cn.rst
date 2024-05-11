.. _cn_api_paddle_linalg_ormqr:

ormqr
-------------------------------

.. py:function:: paddle.linalg.ormqr(x, tau, other, left=True, transpose=False)

计算维度为(m, n)的矩阵 C（由 :attr:`other` 给出）和一个矩阵 Q 的乘积，
其中 Q 由 Householder 反射系数 (:attr:`x`, :attr:`tau`) 表示。

参数
::::::::::::

    - **x** (Tensor) - 输入一个或一批矩阵，类型为 Tensor。 ``x`` 的形状应为 ``[*, MN, K]``，其中 ``*`` 为零或更大的批次维度，数据类型支持 float32， float64。
    - **tau** (Tensor) - 输入一个或一批 Householder 反射系数，类型为 Tensor。 ``tau`` 的形状应为 ``[*, min(MN, K)]``，其中 ``*`` 为零或更大的批次维度，数据类型支持 float32， float64。
    - **other** (Tensor) - 输入一个或一批矩阵，类型为 Tensor。 ``other`` 的形状应为 ``[*, M, N]``，其中 ``*`` 为零或更大的批次维度，数据类型支持 float32， float64。
    - **left** (bool， 可选) - 决定了矩阵乘积运算的顺序。如果 left 为 ``True`` ，计算顺序为 op(Q) * other ，否则，计算顺序为 other * op(Q)。默认值： ``True`` 。
    - **transpose** (bool， 可选) - 如果为 ``True`` ，对矩阵 Q 进行共轭转置变换，否则，不对矩阵 Q 进行共轭转置变换。默认值： ``False`` 。
返回
::::::::::::

    ``Tensor``，维度和数据类型都与 :attr:`other` 一致。

代码示例
::::::::::

COPY-FROM: paddle.linalg.ormqr
