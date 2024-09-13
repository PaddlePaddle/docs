.. _cn_api_paddle_linalg_matrix_rank:

matrix_rank
-------------------------------

.. py:function:: paddle.linalg.matrix_rank(x, tol=None, hermitian=False, atol=None, rtol=None, name=None)


计算矩阵的秩。

.. note::
    1. 阈值参数支持单独使用 ``tol`` 或联合使用 ``atol`` 与 ``rtol``。
    2. 当单独使用 ``tol`` 时：若 hermitian=False ，矩阵的秩是大于指定的 ``tol`` 阈值的奇异值的数量；若 hermitian=True ，矩阵的秩是大于指定 ``tol`` 阈值的特征值绝对值的数量。
    3. 当联合使用 ``atol`` 与 ``rtol`` 时，阈值的计算方式是 ``max(atol, sigma_1 * rtol)`` ，其中 ``sigma_1`` 是奇异值中的最大值（或特征值绝对值的最大值）。
    4. 当联合使用 ``atol`` 与 ``rtol`` 时：若 ``rtol`` 未被声明，同时 ``atol`` 未被声明或取值为 0 ， ``rtol`` 将被设为 ``max(m,n) * eps`` ，其中 ``m`` ， ``n`` 分别是 ``x`` 矩阵的行数和列数， ``eps`` 为 ``x`` 的类型的 epsilon 值；若 ``rtol`` 未被声明，同时 ``atol`` 取值大于 0 ， ``rtol`` 将被设为 0 。

参数
:::::::::
    - **x** (Tensor) - 输入 tensor。它的形状应该是 ``[..., m, n]``，其中 ``...`` 是零或者更大的批次维度。如果 ``x`` 是一批矩阵，则输出具有相同的批次尺寸。``x`` 的数据类型应该为 float32 或 float64。
    - **tol** (float|Tensor，可选) - 阈值。默认值：None。如果未指定 ``tol`` ， ``sigma`` 为所计算奇异值中的最大值（或特征值绝对值的最大值）， ``eps`` 为 ``x`` 的类型的 epsilon 值，使用公式 ``tol=sigma * max(m,n) * eps`` 来计算 ``tol``。请注意，如果 ``x`` 是一批矩阵，以这种方式为每批矩阵计算 ``tol`` 。
    - **hermitian** (bool，可选) - 表示 ``x`` 是否是 Hermitian 矩阵。默认值：False。当 hermitian=True 时，``x`` 被假定为 Hermitian 矩阵，这时函数内会使用更高效的算法来求解特征值，但在函数内部不会对 ``x`` 进行检查。我们仅仅使用矩阵的下三角来进行计算。
    - **atol** (float|Tensor，可选) - 绝对阈值。未被声明时当作 0 。默认值：None。
    - **rtol** (float|Tensor，可选) - 相对阈值。未被声明时取值参照上述 Note（注解） 。默认值：None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
Tensor， ``x`` 的秩，数据类型为 int64。

代码示例
::::::::::

COPY-FROM: paddle.linalg.matrix_rank
