.. _cn_api_paddle_nn_PairwiseDistance:

PairwiseDistance
-------------------------------

.. py:class:: paddle.nn.PairwiseDistance(p=2., epsilon=1e-6, keepdim=False, name=None)

计算两组向量（输入 ``x``、``y``）两两之间的距离。该距离通过 p 范数计算：

.. math::

    \Vert x \Vert _p = \left( \sum_{i=1}^n \vert x_i \vert ^ p \right ) ^ {1/p}.

参数
::::::::
    - **p** (float，可选) - 指定 p 阶的范数。默认值为 :math:`2.0`。
    - **epsilon** (float，可选) - 添加到分母的一个很小值，避免发生除零错误。默认值为 :math:`1e-6`。
    - **keepdim** (bool，可选) - 是否保留输出 Tensor 减少的维度。输出结果相对于 :math:`|x-y|` 的结果减少一维，除非 :attr:`keepdim` 为 True，默认值为 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
::::::::
    - **x** (Tensor) - :math:`[N, D]` 或 :math:`[D]`，其中 :math:`N` 是批大小，:math:`D` 是向量的维度，数据类型为 float16，float32，float64。
    - **y** (Tensor) - :math:`[N, D]` 或 :math:`[D]`，与 :attr:`x` 的数据类型相同。
    - **output** (Tensor) - 数据类型与 :attr:`x`、 :attr:`y` 相同。
        - 如果 :attr:`keepdim` 为 True，则形状为 :math:`[N, 1]` 或 :math:`[1]`，依据输入中是否有数据形状为 :math:`[N, D]`。
        - 如果 :attr:`keepdim` 为 False，则形状为 :math:`[N]` 或 :math:`[]`，依据输入中是否有数据形状为 :math:`[N, D]`。

代码示例
::::::::

COPY-FROM: paddle.nn.PairwiseDistance
