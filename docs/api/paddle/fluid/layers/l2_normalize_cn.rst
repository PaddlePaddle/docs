.. _cn_api_fluid_layers_l2_normalize:

l2_normalize
-------------------------------

.. py:function:: paddle.fluid.layers.l2_normalize(x,axis,epsilon=1e-12,name=None)




该 OP 计算欧几里得距离之和对 x 进行归一化。对于 1-DTensor（系数矩阵的维度固定为 0）
计算公式如下：

.. math::

    y=\frac{x}{\sqrt{\sum x^{2}+epsilon}}

对于输入为多维 Tensor 的情况，该 OP 分别对维度轴上的每个 1-D 切片单独归一化

参数
::::::::::::

    - **x** (Variable) - 维度为 :math:`[N_1, N_2, ..., N_k, D]` 的多维 Tensor，其中最后一维 D 是类别数目。数据类型为 float32 或 float64。
    - **axis** (int) - 归一化的轴。如果轴小于 0，归一化的维是 rank(X)+axis。其中，-1 用来表示最后一维。
    - **epsilon** (float) - epsilon，用于避免除 0，默认值为 1e-12。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

    返回：与输入 x 的维度一致的 Tensor

    返回类型：Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.l2_normalize
