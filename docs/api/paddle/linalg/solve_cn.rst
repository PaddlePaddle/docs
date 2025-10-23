.. _cn_api_paddle_linalg_solve:

solve
-------------------------------

.. py:function:: paddle.linalg.solve(x, y, left=True, name=None)


计算线性方程组的解。

记 :math:`X` 为一个或一批方阵，:math:`Y` 一个或一批矩阵，当 `left` 为 True 时，公式为：

.. math::
    Out = X ^ {-1} * Y

当 `left` 为 False 时，公式为：

.. math::
    Out = Y * X ^ {-1}

特别地，

- 如果 ``X`` 不可逆，则线性方程组不可解。


参数
:::::::::
    - **x** (Tensor) - 输入的欲进行线性方程组求解的一个或一批方阵（系数矩阵），类型为 Tensor。 ``x`` 的形状应为 ``[*, M, M]``，其中 ``*`` 为零或更大的批次维度，数据类型为 float32， float64。
    - **y** (Tensor) - 输入的欲进行线性方程组求解的右值，类型为 Tensor。 ``y`` 的形状应为 ``[*, M, K]``，其中 ``*`` 为零或更大的批次维度，数据类型和 ``x`` 相同。
    - **left** (bool，可选) - 设置所求解的线性方程组是 :math:`X * Out = Y` 或 :math:`Out * X = Y`。默认值为 True，表示所求解的线性方程组是 :math:`X * Out = Y`。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

Tensor，这个（或这批）矩阵 ``x`` 与 ``y`` 经过运算后的结果，数据类型和输入 ``x`` 的一致。

代码示例
::::::::::

COPY-FROM: paddle.linalg.solve
