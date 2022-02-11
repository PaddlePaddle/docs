.. _cn_api_linalg_triangular_solve:

triangular_solve
-------------------------------

.. py:function:: paddle.linalg.triangular_solve(x, y, upper=True, transpose=False, unitriangular=False, name=None)


计算具有唯一解的线性方程组解，其中系数矩阵 `x` 是上(下)三角系数矩阵， `y` 是方程右边。

记 :math:`X` 为一个或一批方阵，:math:`Y` 一个或一批矩阵。

则方程组为：

.. math::
    X * Out = Y

方程组的解为：

.. math::
    Out = X ^ {-1} * Y

特别地，

- 如果 ``x`` 不可逆 ， 则线性方程组不可解。

参数
:::::::::
    - **x** (Tensor) : 线性方程组左边的系数方阵，其为一个或一批方阵。``x`` 的形状应为 ``[*, M, M]``，其中 ``*`` 为零或更大的批次维度，数据类型为float32， float64。
    - **y** (Tensor) : 线性方程组右边的矩阵，其为一个或一批矩阵。``y`` 的形状应为 ``[*, M, K]``， 其中 ``*`` 为零或更大的批次维度，数据类型为float32， float64。
    - **upper** (bool, 可选) - 对系数矩阵 ``x`` 取上三角还是下三角。默认为True，表示取上三角。
    - **transpose** (bool, 可选) - 是否对系数矩阵 ``x`` 进行转置。默认为False，不进行转置。
    - **unitriangular** (bool, 可选) - 如果为True，则将系数矩阵 ``x`` 对角线元素假设为1来求解方程。默认为False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：
:::::::::
    - Tensor， 线程方程组的解， 数据类型和 ``x`` 一致。

代码示例
::::::::::

.. code-block:: python

    # a square system of linear equations:
    # x1 +   x2  +   x3 = 0
    #      2*x2  +   x3 = -9
    #               -x3 = 5

    import paddle
    import numpy as np

    x = paddle.to_tensor([[1, 1, 1], 
                          [0, 2, 1],
                          [0, 0,-1]], dtype="float64")
    y = paddle.to_tensor([[0], [-9], [5]], dtype="float64")
    out = paddle.linalg.triangular_solve(x, y, upper=True)

    print(out)
    # [7, -2, -5]

