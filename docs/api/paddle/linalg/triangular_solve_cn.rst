.. _cn_api_linalg_triangular_solve:

triangular_solve
-------------------------------

.. py:function:: paddle.linalg.triangular_solve(x, y, upper=True, transpose=False, unitriangular=False, name=None)


计算具有唯一解的线性方程组解，其中参数 `x` 是上(下)三角系数矩阵，`y` 为线性方程组右边的矩阵。

记 :math:`x` 表示一个或一批方阵，:math:`y` 表示一个或一批矩阵。

则方程组为：

.. math::
    x * Out = y

方程组的解为：

.. math::
    Out = x ^ {-1} * y

特别地，

- 如果 ``x`` 不可逆，则线性方程组不可解。

参数
:::::::::
    - **x** (Tensor) - 线性方程组左边的系数方阵，其为一个或一批方阵。``x`` 的形状应为 ``[*, M, M]``，其中 ``*`` 为零或更大的批次维度，数据类型为 float32， float64。
    - **y** (Tensor) - 线性方程组右边的矩阵，其为一个或一批矩阵。``y`` 的形状应为 ``[*, M, K]``，其中 ``*`` 为零或更大的批次维度，数据类型为 float32， float64。
    - **upper** (bool，可选) - 对系数矩阵 ``x`` 取上三角还是下三角。默认为 True，表示取上三角。
    - **transpose** (bool，可选) - 是否对系数矩阵 ``x`` 进行转置。默认为 False，不进行转置。
    - **unitriangular** (bool，可选) - 如果为 True，则将系数矩阵 ``x`` 对角线元素假设为 1 来求解方程。默认为 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

Tensor，线程方程组的解，数据类型和 ``x`` 一致。

代码示例
::::::::::

COPY-FROM: paddle.linalg.triangular_solve
