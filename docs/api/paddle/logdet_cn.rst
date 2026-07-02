.. _cn_api_paddle_logdet:

logdet
-------------------------------

.. py:function:: paddle.logdet(x, name=None)

计算一个或一批方阵的行列式的自然对数。

如果行列式值为正数，则返回行列式的自然对数；如果行列式值为负数或零，则返回 ``NaN``。

参数
::::::::::::

    - **x** (Tensor) - 输入一个或批量方阵。``x`` 的形状应为 ``[*, M, M]``，其中 ``*`` 为零或更大的批次维度，数据类型支持 float32、float64、complex64、complex128。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

Tensor，输出矩阵的行列式的自然对数值。

代码示例
::::::::::::

COPY-FROM: paddle.logdet
