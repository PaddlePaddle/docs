.. _cn_api_nn_PairwiseDistance:

PairwiseDistance
-------------------------------

.. py:class:: paddle.nn.PairwiseDistance(p=2., epsilon=1e-6, keepdim=False, name=None)

该OP计算两个向量（输入 ``x``、``y`` ）之间pairwise的距离。该距离通过p范数计算：

    .. math::

            \Vert x \Vert _p = \left( \sum_{i=1}^n \vert x_i \vert ^ p \right ) ^ {1/p}.

参数
::::::::
    - **p** （float，可选）- 指定p阶的范数。默认值为2。
    - **epsilon** （float，可选）- 添加到分母的一个很小值，避免发生除零错误。默认值为1e-6。
    - **keepdim** （bool，可选）- 是否保留输出张量减少的维度。输出结果相对于 ``|x-y|`` 的结果减少一维，除非 :attr:`keepdim` 为True，默认值为False。
    - **name** （str，可选） - 操作的名称（可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

形状
::::::::
    - **x** (Tensor) - :math:`(N, D)` ，其中D是向量的维度，数据类型为float32或float64。
    - **y** (Tensor) - :math:`(N, D)` ，与 ``x`` 的形状、数据类型相同。
    - **output** (Tensor) - :math:`(N)` ，如果 :attr:`keepdim` 为True，则形状为 :math:`(N, 1)` 。数据类型与 ``x``、 ``y`` 相同。

代码示例
::::::::

..  code-block:: python

            import paddle
            paddle.disable_static()
            x = paddle.to_tensor([[1., 3.], [3., 5.]], dtype='float64')
            y = paddle.to_tensor([[5., 6.], [7., 8.]], dtype='float64')
            dist = paddle.nn.PairwiseDistance()
            distance = dist(x, y)
            print(distance.numpy()) # [5. 5.]

