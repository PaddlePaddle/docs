.. _cn_api_paddle_frac:

frac
-------------------------------

.. py:function:: paddle.frac(x, name=None)


得到输入 `Tensor` 的小数部分。


参数
:::::::::
    - **x** (Tensor)：输入变量，类型为 Tensor，支持 int32、int64、float32、float64 数据类型。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
    - Tensor (Tensor)，输入矩阵只保留小数部分的结果。


代码示例
:::::::::

COPY-FROM: paddle.frac
