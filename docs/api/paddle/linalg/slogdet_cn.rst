.. _cn_api_paddle_linalg_slogdet:

slogdet
-------------------------------

.. py:function:: paddle.linalg.slogdet(x)
计算批量矩阵的行列式值的符号值和行列式值绝对值的自然对数值。如果行列式值为 0，则符号值为 0，自然对数值为-inf。

参数
::::::::::::

    - **x** (Tensor)：输入一个或批量矩阵。``x`` 的形状应为 ``[*, M, M]``，其中 ``*`` 为零或更大的批次维度，数据类型支持 float32、float64。

返回
::::::::::::

Tensor，输出矩阵的行列式值 Shape 为 ``[2, *]``。

代码示例
::::::::::

COPY-FROM: paddle.linalg.slogdet
