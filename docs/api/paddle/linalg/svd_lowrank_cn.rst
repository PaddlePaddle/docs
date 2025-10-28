.. _cn_api_paddle_linalg_svd_lowrank:

svd_lowrank
-------------------------------

.. py:function:: paddle.linalg.svd_lowrank(x, q=None, niter=2, M=None, name=None)


计算在低秩矩阵或者批次的矩阵上进行奇异值分解（SVD）。

记 :math:`X` 为一个矩阵或者批次矩阵，输出结果满足：

.. math::
    X \approx U * diag(S) * V^{T}

若提供了 :math:`M` ，输出结果满足：

.. math::
    X - M \approx U * diag(S) * V^{T}

参数
::::::::::::

    - **x** (Tensor) - 输入的需要进行奇异值分解的一个或一批方阵，类型为 Tensor。 ``x`` 的形状应为 ``[*, M, N]``，其中 ``*`` 为零或更大的批次维度，数据类型支持 float32， float64。
    - **q** (int，可选) - 对输入 :math:`X` 的秩稍微高估的预估值，默认值为 None，代表预估值取 6。
    - **niter** (int) - 需要进行的子空间迭代次数。默认值为 2。
    - **M** (Tensor) - 输入矩阵在 `axis=-2` 维上的均值，形状应为 ``[*, 1, N]``，默认为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

    - Tensor U，形状为 N x q 的矩阵。
    - Tensor S，长度为 q 的向量。
    - Tensor V，形状为 M x q 的矩阵。

    tuple (U, S, V): 对输入 :math:`X` 或 :math:`X-M` 的奇异值分解的近似最优解。

代码示例
::::::::::

COPY-FROM: paddle.linalg.svd_lowrank
