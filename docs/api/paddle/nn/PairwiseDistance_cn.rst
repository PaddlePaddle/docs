.. _cn_api_nn_PairwiseDistance:

PairwiseDistance
-------------------------------

.. py:class:: paddle.nn.PairwiseDistance(p=2., epsilon=1e-6, keepdim=False, name=None)

该 API 计算两组向量（输入 ``x``、``y``）之间 pairwise 的距离。该距离通过 p 范数计算：

    .. math::

            \Vert x \Vert _p = \left( \sum_{i=1}^n \vert x_i \vert ^ p \right ) ^ {1/p}.

参数
::::::::
    - **p** (float，可选)- 指定 p 阶的范数。默认值为 2。
    - **epsilon** (float，可选)- 添加到分母的一个很小值，避免发生除零错误。默认值为 1e-6。
    - **keepdim** (bool，可选)- 是否保留输出张量减少的维度。输出结果相对于 ``|x-y|`` 的结果减少一维，除非 :attr:`keepdim` 为 True，默认值为 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
::::::::
    - **x** (Tensor) - :math:`(N, D)`，其中 D 是向量的维度，数据类型为 float32 或 float64。
    - **y** (Tensor) - :math:`(N, D)`，与 ``x`` 的形状、数据类型相同。
    - **output** (Tensor) - :math:`(N)`，如果 :attr:`keepdim` 为 True，则形状为 :math:`(N, 1)`。数据类型与 ``x``、 ``y`` 相同。

代码示例
::::::::

COPY-FROM: paddle.nn.PairwiseDistance
