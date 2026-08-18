.. _cn_api_paddle_distribution_constraint_positive_definite_check:

positive_definite.check
-------------------------------

.. py:method:: paddle.distribution.constraint.positive_definite.check(value)

检查输入的最后两个维度是否构成对称正定矩阵。

参数
:::::::::

    - **value** (Tensor) - 要检查的 Tensor；最后两个维度表示矩阵。

返回
:::::::::

Tensor，bool 类型，形状为 ``value.shape[:-2]``，表示每个矩阵是否为对称正定矩阵。

