.. _cn_api_linalg_cholesky:

cholesky
-------------------------------

.. py:function:: paddle.linalg.cholesky(x, upper=False, name=None)




计算一个对称正定矩阵或一批对称正定矩阵的 Cholesky 分解。如果 `upper` 是 `True`，
则分解形式为 :math:`A = U ^ {T} U`，返回的矩阵 U 是上三角矩阵。
否则，分解形式为 :math:`A = LL ^ {T}`，并返回矩阵 :math:`L` 是下三角矩阵。

参数
::::::::::::

    - **x** （Tensor）- 输入变量为多维 Tensor，它的维度应该为 `[*, M, N]`，其中*为零或更大的批次尺寸，并且最里面的两个维度上的矩阵都应为对称的正定矩阵，支持数据类型为 float32、float64。
    - **upper** （bool）- 指示是否返回上三角矩阵或下三角矩阵。默认值：False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，与 `x` 具有相同形状和数据类型。它代表了 Cholesky 分解生成的三角矩阵。

代码示例
::::::::::::

COPY-FROM: paddle.linalg.cholesky
