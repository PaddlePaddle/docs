.. _cn_api_paddle_mv:

mv
-------------------------------

.. py:function:: paddle.mv(x, vec, name=None)

计算矩阵 ``x`` 和向量 ``vec`` 的乘积。

参数
:::::::::
    - **x** (Tensor) - 输入变量，类型为 Tensor，形状为 :math:`[M, N]`，数据类型为 float32、float64。
    - **vec** (Tensor) - 输入变量，类型为 Tensor，形状为 :math:`[N]`，数据类型为 float32、float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::

    - Tensor，矩阵 ``x`` 和向量 ``vec`` 的乘积。

代码示例
::::::::::

COPY-FROM: paddle.mv
