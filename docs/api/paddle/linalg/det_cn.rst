.. _cn_api_paddle_linalg_det:

det
-------------------------------

.. py:function:: paddle.linalg.det(x, name=None, *, out=None)

计算一个或批量方阵的行列式值。

参数
::::::::::::

    - **x** (Tensor)：输入一个或批量方阵。``x`` 的形状应为 ``[*, n, n]``，其中 ``*`` 为零或更大的批次维度。别名 ``input``。
    - **name** (str|None，可选) - 该参数用于打印调试信息，具体用法请参见 :ref:`api_guide_Name`，默认值为 None。

关键字参数
::::::::::::
    - **out** (Tensor，可选) - 指定输出的存储结果的 Tensor，如果提供，返回值将是 ``out`` 本身。默认值为 ``None``。

返回
::::::::::::

Tensor，输出方阵的行列式值。Shape 为 ``[*]`` 。

代码示例
::::::::::

COPY-FROM: paddle.linalg.det
