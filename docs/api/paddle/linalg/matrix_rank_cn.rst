.. _cn_api_linalg_matrix_rank:

matrix_rank
-------------------------------

.. py:function:: paddle.linalg.matrix_rank(x, tol=None, hermitian=False, name=None)


计算矩阵的秩

当hermitian=False时，矩阵的秩是大于指定的tol阈值的奇异值的数量；当hermitian=True时，矩阵的秩是大于指定tol阈值的特征值绝对值的数量。


参数：
    - **x** (Tensor): 输入tensor。
        它的形状应该是''[..., m, n]''，其中...是零或多更大的批次维度。如果x是一批矩阵，则输出具有相同的批次尺寸。x的数据类型应该为float32或float64。
    - **tol** (float, Tensor, 可选): 阈值。默认值：None。
        如果未指定tol，sigma为所计算奇异值中的最大值（或特征值绝对值的最大值），eps为x的类型的epsilon值，使用公式tol=sigma * max(m,n) * eps来计算tol。
        请注意，如果x是一批矩阵，以这种方式为每批矩阵计算tol。
    - **hermitian** (Tensor): 表示x是否是Hermitian矩阵。 默认值：False。
        当hermitian=True时，x被假定为Hermitian矩阵，但在函数内部不会对x进行检查。我们仅仅使用矩阵的下三角来进行计算。
    - **name** (str, 可选): OP的名称，默认值：None. 更多信息请参考: ref:`api_guide_Name`.


返回：
    - Tensor: x的秩.

**代码示例**：

.. code-block:: python

    import paddle
    a = paddle.eye(10)
    b = paddle.linalg.matrix_rank(a)
    print(b)
    # b = [10]

    c = paddle.ones(shape=[3, 4, 5, 5])
    d = paddle.linalg.matrix_rank(c, tol=0.01, hermitian=True)
    print(d)
    # d = [[1, 1, 1, 1],
    #      [1, 1, 1, 1],
    #      [1, 1, 1, 1]]