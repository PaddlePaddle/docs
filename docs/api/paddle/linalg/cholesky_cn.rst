.. _cn_api_linalg_cholesky:

cholesky
-------------------------------

.. py:function:: paddle.linalg.cholesky(x, upper=False, name=None)




计算一个对称正定矩阵或一批对称正定矩阵的Cholesky分解。如果 `upper` 是 `True` ，
则分解形式为 :math:`A = U ^ {T} U` , 返回的矩阵U是上三角矩阵。
否则，分解形式为 :math:`A = LL ^ {T}` ，并返回矩阵 :math:`L` 是下三角矩阵。

参数：
    - **x** （Tensor）- 输入变量为多维Tensor，它的维度应该为 `[*, M, N]` ,其中*为零或更大的批次尺寸，并且最里面的两个维度上的矩阵都应为对称的正定矩阵，支持数据类型为float32，float64。
    - **upper** （bool）- 指示是否返回上三角矩阵或下三角矩阵。默认值：False。
    - **name** （str ， 可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回： 与 `x` 具有相同形状和数据类型的Tensor。它代表了Cholesky分解生成的三角矩阵。

返回类型：  变量（Tensor）

**代码示例**

..  code-block:: python

    import paddle
    import numpy as np

    a = np.random.rand(3, 3)
    a_t = np.transpose(a, [1, 0])
    x_data = np.matmul(a, a_t) + 1e-03
    x = paddle.to_tensor(x_data)
    out = paddle.linalg.cholesky(x, upper=False)
    print(out)
    # [[1.190523   0.         0.        ]
    #  [0.9906703  0.27676893 0.        ]
    #  [1.25450498 0.05600871 0.06400121]]
