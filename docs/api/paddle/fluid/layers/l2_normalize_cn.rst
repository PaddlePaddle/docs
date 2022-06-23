.. _cn_api_fluid_layers_l2_normalize:

l2_normalize
-------------------------------

.. py:function:: paddle.fluid.layers.l2_normalize(x,axis,epsilon=1e-12,name=None)




该OP计算欧几里得距离之和对x进行归一化。对于1-D张量（系数矩阵的维度固定为0）
计算公式如下：

.. math::

    y=\frac{x}{\sqrt{\sum x^{2}+epsilon}}

对于输入为多维Tensor的情况，该OP分别对维度轴上的每个1-D切片单独归一化

参数
::::::::::::

    - **x** (Variable) - 维度为 :math:`[N_1, N_2, ..., N_k, D]` 的多维Tensor，其中最后一维D是类别数目。数据类型为float32或float64。
    - **axis** (int) - 归一化的轴。如果轴小于0，归一化的维是rank(X)+axis。其中，-1用来表示最后一维。
    - **epsilon** (float) - epsilon，用于避免除0，默认值为1e-12。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

    返回：与输入x的维度一致的Tensor

    返回类型：Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.l2_normalize