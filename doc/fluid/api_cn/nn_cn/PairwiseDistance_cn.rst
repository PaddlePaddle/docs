.. _cn_api_paddle_nn_PairwiseDistance:

PairwiseDistance
-------------------------------

.. py:class:: paddle.nn.PairwiseLoss(p=2., eps=1e-6, keepdim=False)

:alias_main: paddle.nn.PairwiseDistance
:alias: paddle.nn.PairwiseDistance,paddle.nn.layer.PairwiseDistance,paddle.nn.layer.distance.PairwiseDistance




该OP计算两个向量（输入x、y）之间pairwise的距离。该距离通过p范数计算：

    .. math::

            \Vert x \Vert _p = \left( \sum_{i=1}^n \vert x_i \vert ^ p \right ) ^ {1/p}.

参数：
    - **p** (float, 可选): - 指定p阶的范数。默认值为2。
    - **eps** (float, 可选): - 添加一个很小的值，避免发生除零错误。默认值为1e-6。
    - **keepdim** (bool, 可选): - 是否保留输出张量减少的维度。输出结果相对于`|x-y|`的结果减少一维，除非 :attr:`keepdim` 为True，默认值为False。

形状：
    - **x** (Tensor): - :math:`(N, D)` ，其中D是向量的维度，数据类型为float32或float64。
    - **y** (Tensor): - :math:`(N, D)` ，与`x`的形状、数据类型相同。
    - **out** (Tensor): - :math:`(N)` ，如果 :attr:`keepdim` 为True，则形状为 :math:`(N, 1)` 。数据类型与x、y相同。

**代码示例**

..  code-block:: python

            import paddle
            import numpy as np
            paddle.enable_imperative()
            x_np = np.array([[1., 3.], [3., 5.]]).astype(np.float64)
            y_np = np.array([[5., 6.], [7., 8.]]).astype(np.float64)
            x = paddle.imperative.to_variable(x_np)
            y = paddle.imperative.to_variable(y_np)
            dist = paddle.nn.PairwiseDistance()
            distance = dist(x, y)
            print(distance.numpy()) # [5. 5.]

