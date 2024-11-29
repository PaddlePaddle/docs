.. _cn_api_paddle_linalg_svdvals: 

svdvals 
------------------------------- 

.. py:function:: paddle.linalg.svdvals(x, name=None) 

计算一个或一批矩阵的奇异值。 

记 :math:`X` 为输入的矩阵或一批矩阵，则输出的奇异值 :math:`S`是奇异值分解后矩阵的的对角元素： 

.. math:: 
    X = U * diag(S) * VH 
    
值得注意的是，:math:`S`  是一个向量，其元素按从大到小的顺序排列，表示每个奇异值。 


参数 
::::::::::::
    
    - **x** (Tensor) - 输入的欲进行奇异值分解的一个或一批矩阵，类型为 Tensor。 ``x`` 的形状应为 ``[*, M, N]``，其中 ``*`` 为零或更大的批次维度，数据类型支持 float32， float64。 
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。 
    
返回 
::::::::::::

    - Tensor S，奇异值向量，Shape 为 ``[*, K]`` ，其中 K 为 M 和 N 的最小值。 
    
代码示例 
::::::::::

COPY-FROM: paddle.linalg.svdvals