.. _cn_api_paddle_sparse_pca_lowrank:

pca_lowrank
-------------------------------

.. py:function:: paddle.sparse.pca_lowrank(x, q=None, center=True, niter=2, name=None)


计算在稀疏矩阵上的线性主成分分析(PCA)。

记 :math:`X` 为一个稀疏矩阵，输出结果满足：

.. math::
    X = U * diag(S) * V^{T}


参数
::::::::::::

    - **x** (Tensor) - 输入的需要进行线性主成分分析的一个稀疏方阵，类型为 Tensor。 ``x`` 的形状应为 ``[M, N]``，数据类型支持 float32， float64。
    - **q** (int，可选) - 对输入 :math:`X` 的秩稍微高估的预估值，默认值是 :math:`q=min(6,N,M)`。
    - **center** (bool，可选) - 是否对输入矩阵进行中心化操作，类型为 bool ，默认为 True。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

    - Tensor U，形状为 N x q 的矩阵。
    - Tensor S，长度为 q 的向量。
    - Tensor V，形状为 M x q 的矩阵。

    tuple (U, S, V): 对输入 :math:`X` 的奇异值分解的近似最优解。

代码示例
::::::::::

COPY-FROM: paddle.sparse.pca_lowrank
