.. _cn_api_paddle_linalg_ormqr:

ormqr
-------------------------------

.. py:function:: paddle.linalg.ormqr(x, tau, y, left=True, transpose=False, name=None)

计算维度为(m, n)的矩阵 C（由 :attr:`y` 给出）和一个矩阵 Q 的乘积，
其中 Q 由 Householder 反射系数 (:attr:`x`, :attr:`tau`) 表示。

参数
::::::::::::

    - **x** (Tensor) - 输入一个或一批矩阵，类型为 Tensor。 ``x`` 的形状应为 ``[*, MN, K]``，其中 ``*`` 为零或更大的批次维度，数据类型支持 float16、float32、float64、complex64、complex128。
    - **tau** (Tensor) - 输入一个或一批 Householder 反射系数，类型为 Tensor。 ``tau`` 的形状应为 ``[*, min(MN, K)]``，其中 ``*`` 为零或更大的批次维度，数据类型与 ``x`` 相同，支持 float16、float32、float64、complex64、complex128。
    - **y** (Tensor) - 输入一个或一批矩阵，类型为 Tensor。 ``y`` 的形状应为 ``[*, M, N]``，其中 ``*`` 为零或更大的批次维度，数据类型与 ``x`` 相同，支持 float16、float32、float64、complex64、complex128。
    - **left** (bool， 可选) - 决定了矩阵乘积运算的顺序。如果 left 为 ``True`` ，计算顺序为 op(Q) * y ，否则，计算顺序为 y * op(Q)。默认值： ``True`` 。
    - **transpose** (bool， 可选) - 如果为 ``True`` ，对矩阵 Q 进行共轭转置变换，否则，不对矩阵 Q 进行共轭转置变换。默认值： ``False`` 。
    - **name** (str|None，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
返回
::::::::::::

    ``Tensor``，维度和数据类型都与 :attr:`y` 一致。

代码示例
::::::::::

COPY-FROM: paddle.linalg.ormqr
