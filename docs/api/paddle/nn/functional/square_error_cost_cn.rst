.. _cn_api_fluid_layers_square_error_cost:

square_error_cost
-------------------------------

.. py:function:: paddle.nn.functional.square_error_cost(input,label)


用于计算预测值和目标值的方差估计。

对于预测值 input 和目标值 label，公式为：

.. math::

    Out = (input-label)^{2}

参数
::::::::::::

    - **input** (Tensor) - 预测值，维度为 :math:`[N_1, N_2, ..., N_k, D]` 的多维 Tensor，其中最后一维 D 是类别数目。数据类型为 float32 或 float64。
    - **label** (Tensor) - 目标值，维度为 :math:`[N_1, N_2, ..., N_k, D]` 的多维 Tensor，其中最后一维 D 是类别数目。数据类型为 float32 或 float64。

返回
::::::::::::
Tensor，预测值和目标值的方差


代码示例
::::::::::::

COPY-FROM: paddle.nn.functional.square_error_cost
