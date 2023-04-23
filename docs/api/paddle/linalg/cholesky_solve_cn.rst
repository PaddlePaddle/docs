.. _cn_api_linalg_cholesky_solve:

cholesky_solve
-------------------------------

.. py:function:: paddle.linalg.cholesky_solve(x, y, upper=False, name=None)

对 A @ X = B 的线性方程求解，其中 A 是方阵，输入 x、y 分别是矩阵 B 和矩阵A 的 Cholesky 分解矩阵 u。

输入 x、y 是 2 维矩阵，或者 2 维矩阵以 batch 形式组成的 3 维矩阵。如果输入是 batch 形式的 3 维矩阵，则输出也是 batch 形式的 3 维矩阵。

参数
::::::::::::

    - **x** (Tensor) - 线性方程中的 B 矩阵。是 2 维矩阵或者 2 维矩阵以 batch 形式组成的 3 维矩阵。
    - **y** (Tensor) - 线性方程中 A 矩阵的 Cholesky 分解矩阵 u，上三角或者下三角矩阵。是 2 维矩阵或者 2 维矩阵以 batch 形式组成的 3 维矩阵。
    - **upper** (bool，可选) - 输入 x 是否是上三角矩阵，True 为上三角矩阵，False 为下三角矩阵。默认值 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，线性方程的解 X。

代码示例
::::::::::

COPY-FROM: paddle.linalg.cholesky_solve
