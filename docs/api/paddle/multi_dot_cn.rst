.. _cn_api_tensor_multi_dot:

multi_dot
-------------------------------

.. py:function:: paddle.multi_dot(x, name=None)

Multi_dot是一个计算多个矩阵乘法的算子。

算子支持float，double和float16三种类型。该算子不支持批量输入。

输入[x]的每个tensor的shape必须是二维的，除了第一个和做后一个tensor可以是一维的。如果第一个tensor是shape为(n, )的一维向量，该tensor将被当作是shape为(1, n)的行向量处理，同样的，如果最后一个tensor的shape是(n, )，将被当作是shape为(n, 1)的列向量处理。

如果第一个和最后一个tensor是二维矩阵，那么输出也是一个二维矩阵，否则输出是一维的向量。

Multi_dot会选择计算量最小的乘法顺序进行计算。(a, b)和(b, c)这样两个矩阵相乘的计算量是a * b * c。给定矩阵A, B, C的shape分别为(20, 5)， (5, 100)，(100, 10)，我们可以计算不同乘法顺序的计算量：

- Cost((AB)C) = 20x5x100 + 20x100x10 = 30000
- Cost(A(BC)) = 5x100x10 + 20x5x10 = 6000

在这个例子中，先算B乘以C再乘A的计算量比按顺序乘少5被。

参数
:::::::::
    - **x** ([tensor]): 输出的是一个tensor列表。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：
:::::::::
    - Tensor，输出Tensor

代码示例
::::::::::

.. code-block:: python
    import paddle
    import numpy as np
    # A * B
    A_data = np.random.random([3, 4]).astype(np.float32)
    B_data = np.random.random([4, 5]).astype(np.float32)
    A = paddle.to_tensor(A_data)
    B = paddle.to_tensor(B_data)
    out = paddle.multi_dot([A, B])
    print(out.numpy().shape)
    # [3, 5]
    # A * B * C
    A_data = np.random.random([10, 5]).astype(np.float32)
    B_data = np.random.random([5, 8]).astype(np.float32)
    C_data = np.random.random([8, 7]).astype(np.float32)
    A = paddle.to_tensor(A_data)
    B = paddle.to_tensor(B_data)
    C = paddle.to_tensor(C_data)
    out = paddle.multi_dot([A, B, C])
    print(out.numpy().shape)
    # [10, 7]
