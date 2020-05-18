cholesky
-------------------------------

.. py:function:: paddle.cholesky(input, upper=False)

:alias_main: paddle.cholesky
:alias: paddle.cholesky,paddle.tensor.cholesky,paddle.tensor.linalg.cholesky


为单个对称正定矩阵或batch形式的多个对称正定矩阵进行Cholesky分解。

若 :attr:`upper` 为 True，分解的形式为 :math:`A = U^{T}U` 并返回上三角矩阵 :math:`U` ；否则，分解的形式为 :math:`A = LL^{T}` 并返回下三角矩阵 :math:`L` 。

**参数**：
  - **x** (Variable) – 输入张量，其形状为 `[*, M, M]`，其中*表示0个或者多个batch的维度；最内两维上的矩阵都应是对称正定矩阵。支持的数据类型：float32，float64。
  - **upper** (bool，可选) – 指示返回上三角形式还是下三角形式的矩阵。默认值：False。

**返回**：和 :code:`x` 具有相同形状和数据类型的张量，表示Cholesky分解得到的结果。

返回类型：Variable

抛出异常：
    - :code:`TypeError` ，:code:`x` 不是Variable类型，或者数据类型不是float32、float64时
    - :code:`TypeError` ，:code:`upper` 不是bool类型时

**代码示例**：

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np

    with fluid.dygraph.guard():
        a = np.random.rand(3, 3)
        a_t = np.transpose(a, [1, 0])
        x = np.matmul(a, a_t) + 1e-03
        x = fluid.dygraph.to_variable(x)
        out = paddle.cholesky(x, upper=False)
        print(out.numpy())
        # [[1.190523   0.         0.        ]
        #  [0.9906703  0.27676893 0.        ]
        #  [1.25450498 0.05600871 0.06400121]]

