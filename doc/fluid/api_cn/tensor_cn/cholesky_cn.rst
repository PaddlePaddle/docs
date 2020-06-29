.. _cn_api_tensor_cholesky:

cholesky
-------------------------------

.. py:function:: paddle.cholesky(x, upper=False, name=None)

:alias_main: paddle.cholesky
:alias: paddle.cholesky,paddle.tensor.cholesky,paddle.tensor.linalg.cholesky



计算一个对称正定矩阵或一批对称正定矩阵的Cholesky分解。如果`upper`是`True`，
则分解形式为 :math:`A = U ^ {T} U` , 返回的矩阵U是上三角矩阵。
否则，分解形式为 :math:`A = LL ^ {T}` ，并返回矩阵 :math:`L` 是下三角矩阵。

参数：
    - **x** （Variable）- 输入变量为多维Tensor或LoDTensor，它的维度应该为 `[*, M, N]` ,其中*为零或更大的批次尺寸，并且最里面的两个维度上的矩阵都应为对称的正定矩阵，支持数据类型为float32，float64。
    - **upper** （bool）- 指示是否返回上三角矩阵或下三角矩阵。默认值：False。
    - **name** （str ， 可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回： 与`x`具有相同形状和数据类型的Tensor。它代表了Cholesky分解生成的三角矩阵。

返回类型：  变量（Variable）

**代码示例**

..  code-block:: python

      import paddle
      import paddle.fluid as fluid
      import numpy as np

      with fluid.dygraph.guard():
          a = np.random.rand(3, 3)
          print(a)
          # [[0.10548146 0.44426157 0.85944377]
          #  [0.84469568 0.72855948 0.44987977]
          #  [0.34449094 0.89552855 0.79255662]]
          a_t = np.transpose(a, [1, 0])
          x = np.matmul(a, a_t) + 1e-03 * np.eye(3)
          x = fluid.dygraph.to_variable(x)
          out = paddle.cholesky(x, upper=False)
          # [[0.97372392 0.         0.        ]
          #  [0.82098946 0.87958958 0.        ]
          #  [1.1454419  0.40882168 0.26574251]]
