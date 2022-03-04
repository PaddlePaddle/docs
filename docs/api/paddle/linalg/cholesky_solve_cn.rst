.. _cn_api_linalg_cholesky_solve:

cholesky_solve
-------------------------------

.. py:function:: paddle.linalg.cholesky_solve(x, y, upper=False, name=None)

对 A @ X = B 的线性方程求解，其中A是方阵，输入x、y分别是矩阵B和矩阵A的Cholesky分解矩阵u。

输入x、y是2维矩阵，或者2维矩阵以batch形式组成的3维矩阵。如果输入是batch形式的3维矩阵，则输出也是batch形式的3维矩阵。

参数：
:::::::::
    - **x** (Tensor) - 线性方程中的B矩阵。是2维矩阵或者2维矩阵以batch形式组成的3维矩阵。
    - **y** (Tensor) - 线性方程中A矩阵的Cholesky分解矩阵u，上三角或者下三角矩阵。是2维矩阵或者2维矩阵以batch形式组成的3维矩阵。
    - **upper** (bool, 可选) - 输入x是否是上三角矩阵，True为上三角矩阵，False为下三角矩阵。默认值False。
    - **name** (str, 可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：
:::::::::
    - Tensor, 线性方程的解X。

代码示例：
::::::::::

.. code-block:: python

    import paddle

    u = paddle.to_tensor([[1, 1, 1], 
                            [0, 2, 1],
                            [0, 0,-1]], dtype="float64")
    b = paddle.to_tensor([[0], [-9], [5]], dtype="float64")
    out = paddle.linalg.cholesky_solve(b, u, upper=True)

    print(out)
    # [-2.5, -7, 9.5]
