.. _cn_api_linalg_eig:

eig
-------------------------------

.. py:function:: paddle.linalg.eig(x, name=None)

计算一般方阵 ``x`` 的的特征值和特征向量。

.. note::
    - 如果输入矩阵 ``x`` 为 Hermitian 矩阵或实对称阵，请使用更快的 API :ref:`cn_api_linalg_eigh` 。
    - 如果只计算特征值，请使用 :ref:`cn_api_linalg_eigvals` 。
    - 如果矩阵 ``x`` 不是方阵，请使用 :ref:`cn_api_linalg_svd` 。
    - 该 API 当前只能在 CPU 上执行。
    - 对于输入是实数和复数类型，输出的数据类型均为复数。

参数
::::::::::::

    - **x** (Tensor) - 输入一个或一批矩阵。``x`` 的形状应为 ``[*, M, M]``，数据类型支持 float32、float64、complex64 和 complex128。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

    - Tensor Eigenvalues，输出 Shape 为 ``[*, M]`` 的矩阵，表示特征值。
    - Tensor Eigenvectors，输出 Shape 为 ``[*, M, M]`` 矩阵，表示特征向量。

代码示例
::::::::::

COPY-FROM: paddle.linalg.eig
