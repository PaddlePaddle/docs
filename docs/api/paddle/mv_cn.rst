.. _cn_api_paddle_mv:

mv
-------------------------------

.. py:function:: paddle.mv(x, vec, name=None, *, out=None)

计算矩阵 ``x`` 和向量 ``vec`` 的乘积。

参数
:::::::::
    - **x** (Tensor) - 输入变量，类型为 Tensor，形状为 :math:`[M, N]`，数据类型为 float32、float64。别名 ``input``。
    - **vec** (Tensor) - 输入变量，类型为 Tensor，形状为 :math:`[N]`，数据类型为 float32、float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
:::::::::
    - **out** (Tensor，可选) - 输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 中，默认值为 ``None``。

返回
:::::::::

    - Tensor，矩阵 ``x`` 和向量 ``vec`` 的乘积。

代码示例
::::::::::

COPY-FROM: paddle.mv
