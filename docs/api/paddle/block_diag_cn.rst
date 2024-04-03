.. _cn_api_paddle_block_diag:

block_diag
-------------------------------

.. py:function:: paddle.block_diag(inputs, name=None)

根据 `inputs` 创建对角矩阵。

参数
:::::::::
    - **inputs**  (list|tuple) - 是一个 Tensor 列表或 Tensor 元组，其子项为0、1、2维的 Tensor 。数据类型为： `bool`、 `float16`、 `float32`、 `float64`、 `uint8`、 `int8`、 `int16`、 `int32`、 `int64`、 `bfloat16`、 `complex64`、 `complex128`。
    - **name**  (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 `None`。

返回
:::::::::

Tensor， 与 ``inputs`` 数据类型相同。

代码示例
:::::::::

COPY-FROM: paddle.block_diag
