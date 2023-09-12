.. _cn_api_paddle_linalg_matrix_rank:

matrix_rank
-------------------------------

.. py:function:: paddle.linalg.matrix_rank(x, tol=None, hermitian=False, name=None)


计算矩阵的秩。

当 hermitian=False 时，矩阵的秩是大于指定的 ``tol`` 阈值的奇异值的数量；当 hermitian=True 时，矩阵的秩是大于指定 ``tol`` 阈值的特征值绝对值的数量。

参数
:::::::::
    - **x** (Tensor) - 输入 tensor。它的形状应该是 ``[..., m, n]``，其中 ``...`` 是零或者更大的批次维度。如果 ``x`` 是一批矩阵，则输出具有相同的批次尺寸。``x`` 的数据类型应该为 float32 或 float64。
    - **tol** (float|Tensor，可选) - 阈值。默认值：None。如果未指定 ``tol`` ， ``sigma`` 为所计算奇异值中的最大值（或特征值绝对值的最大值）， ``eps`` 为 ``x`` 的类型的 epsilon 值，使用公式 ``tol=sigma * max(m,n) * eps`` 来计算 ``tol``。请注意，如果 ``x`` 是一批矩阵，以这种方式为每批矩阵计算 ``tol`` 。
    - **hermitian** (bool，可选) - 表示 ``x`` 是否是 Hermitian 矩阵。默认值：False。当 hermitian=True 时，``x`` 被假定为 Hermitian 矩阵，这时函数内会使用更高效的算法来求解特征值，但在函数内部不会对 ``x`` 进行检查。我们仅仅使用矩阵的下三角来进行计算。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
Tensor， ``x`` 的秩，数据类型为 int64。

代码示例
::::::::::

COPY-FROM: paddle.linalg.matrix_rank
