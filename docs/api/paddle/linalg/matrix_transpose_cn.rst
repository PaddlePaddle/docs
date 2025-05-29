.. _cn_api_paddle_linalg_matrix_transpose:

matrix_transpose
-------------------------------

.. py:function:: paddle.linalg.matrix_transpose(x, name=None)

对输入张量 `x` 的最后两个维度进行转置。

.. note::
       如果 `n` 是 `x` 的维数，则 `paddle.linalg.matrix_transpose(x)` 等同于 `x.transpose([0, 1, ..., n-2, n-1])` 。

参数
::::::::::::

    - **x** (Tensor) - 输入 Tensor，数据类型为 float32，float64，int32 或者 int64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

 `Tensor`，数据类型与 `x` 相同。

代码示例
::::::::::::

COPY-FROM: paddle.linalg.matrix_transpose
