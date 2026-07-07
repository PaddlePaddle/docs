.. _cn_api_paddle_logdet:

logdet
-------------------------------

.. py:function:: paddle.logdet(input, name=None)

计算一个或一批方阵的行列式的自然对数。

对于行列式为负数的矩阵，返回 ``nan``。
对于行列式为零的矩阵，返回 ``-inf``。

参数
::::::::::::

    - **input** (Tensor) - 输入一个或批量方阵。``input`` 的形状应为 ``[*, n, n]``，其中 ``*`` 为零或更大的批次维度。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

Tensor：输出矩阵的行列式的自然对数值，Shape 为 ``[*]``。

代码示例
::::::::::::

COPY-FROM: paddle.logdet
