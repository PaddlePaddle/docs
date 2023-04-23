.. _cn_api_linalg_det:

det
-------------------------------

.. py:function:: paddle.linalg.det(x)
计算批量矩阵的行列式值。

参数
::::::::::::

    - **x** (Tensor)：输入一个或批量矩阵。``x`` 的形状应为 ``[*, M, M]``，其中 ``*`` 为零或更大的批次维度，数据类型支持 float32、float64。

返回
::::::::::::

Tensor，输出矩阵的行列式值 Shape 为 ``[*]`` 。

代码示例
::::::::::

COPY-FROM: paddle.linalg.det
