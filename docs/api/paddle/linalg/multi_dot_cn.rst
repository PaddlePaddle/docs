.. _cn_api_linalg_multi_dot:

multi_dot
-------------------------------

.. py:function:: paddle.linalg.multi_dot(x, name=None)

Multi_dot 是一个计算多个矩阵乘法的算子。

算子支持 float16(仅限 GPU)、float32 和 float64 三种类型。该算子不支持批量输入。

输入[x]的每个 tensor 的 shape 必须是二维的，除了第一个和最后一个 tensor 可以是一维的。如果第一个 tensor 是 shape 为(n, )的一维向量，该 tensor 将被当作是 shape 为(1, n)的行向量处理，同样的，如果最后一个 tensor 的 shape 是(n, )，将被当作是 shape 为(n, 1)的列向量处理。

如果第一个和最后一个 tensor 是二维矩阵，那么输出也是一个二维矩阵，否则输出是一维的向量。

Multi_dot 会选择计算量最小的乘法顺序进行计算。(a, b)和(b, c)这样两个矩阵相乘的计算量是 a * b * c。给定矩阵 A, B, C 的 shape 分别为(20, 5)， (5, 100)，(100, 10)，我们可以计算不同乘法顺序的计算量：

- Cost((AB)C) = 20x5x100 + 20x100x10 = 30000
- Cost(A(BC)) = 5x100x10 + 20x5x10 = 6000

在这个例子中，先算 B 乘以 C 再乘 A 的计算量比按顺序乘少 5 倍。

参数
:::::::::
    - **x** ([tensor])：输入的是一个 tensor 列表。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor

代码示例
::::::::::

COPY-FROM: paddle.linalg.multi_dot
