.. _cn_api_paddle_linalg_inv:

inv
-------------------------------

.. py:function:: paddle.linalg.inv(x, name=None, *, out=None)


计算方阵的逆。方阵是行数和列数相等的矩阵。输入可以是一个方阵（2-D Tensor），或者是批次方阵（维数大于 2 时）。

参数
:::::::::
  - **x** (Tensor) – 输入 Tensor，最后两维的大小必须相等。如果输入 Tensor 的维数大于 2，则被视为 2-D 矩阵的批次（batch）。支持的数据类型：float32、float64、complex64、complex128。别名 ``input``。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
:::::::::::
    - **out** (Tensor，可选) - 输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 中，默认值为 ``None``。

返回
::::::::
Tensor，输入方阵的逆。


代码示例
:::::::::

COPY-FROM: paddle.linalg.inv
