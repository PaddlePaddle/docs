.. _cn_api_linalg_svd:

svd
-------------------------------

.. py:function:: paddle.linalg.svd(x, full_matrices=False, name=None)


计算一个或一批矩阵的奇异值分解。

记 :math:`X` 为一个矩阵，则计算的结果为 2 个矩阵 :math:`U`, :math:`VH` 和一个向量 :math:`S`。则分解后满足公式：

.. math::
    X = U * diag(S) * VH

值得注意的是，:math:`S` 是向量，从大到小表示每个奇异值。而 :math:`VH` 则是 V 的共轭转置。


参数
::::::::::::

    - **x** (Tensor) - 输入的欲进行奇异值分解的一个或一批方阵，类型为 Tensor。 ``x`` 的形状应为 ``[*, M, N]``，其中 ``*`` 为零或更大的批次维度，数据类型支持 float32， float64。
    - **full_matrices** (bool) - 是否计算完整的 U 和 V 矩阵，类型为 bool 默认为 False。这个参数会影响 U 和 V 生成的 Shape。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

    - Tensor U，奇异值分解的 U 矩阵。如果 full_matrics 设置为 False，则 Shape 为 ``[*, M, K]``，如果 full_metrices 设置为 True，那么 Shape 为 ``[*, M, M]``。其中 K 为 M 和 N 的最小值。
    - Tensor S，奇异值向量，Shape 为 ``[*, K]`` 。
    - Tensor VH，奇异值分解的 VH 矩阵。如果 full_matrics 设置为 False，则 Shape 为 ``[*, K, N]``，如果 full_metrices 设置为 True，那么 Shape 为 ``[*, N, N]``。其中 K 为 M 和 N 的最小值。

代码示例
::::::::::

COPY-FROM: paddle.linalg.svd
