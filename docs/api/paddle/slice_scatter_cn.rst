.. _cn_api_paddle_slice_scatter:

slice_scatter
-------------------------------

.. py:function:: paddle.slice_scatter(x, value, axis=0, start=None, stop=None, step=1, name=None)

将 ``value`` 矩阵的值嵌入到 ``x`` 矩阵的第 ``axis`` 维。返回一个新的 Tensor 而不是试图。

参数
:::::::::
    - **x**  (Tensor) - 输入的 Tensor 作为目标矩阵，数据类型为： `bool`、 `float16`、 `float32`、 `float64`、 `uint8`、 `int8`、 `int16`、 `int32`、 `int64`、 `bfloat16`、 `complex64`、 `complex128`。
    - **value**  (Tensor) - 需要插入的值，数据类型为： `bool`、 `float16`、 `float32`、 `float64`、 `uint8`、 `int8`、 `int16`、 `int32`、 `int64`、 `bfloat16`、 `complex64`、 `complex128`。
    - **axis**  (int) - 指定沿着哪个维度嵌入对应的值。默认为 `0`。
    - **start**  (int，可选) - 嵌入的起始索引。默认为 `None`， `None` 会转换为 `0`。
    - **stop**  (int，可选) - 嵌入的截止索引。默认为 `None`， `None` 会转换为 `x.shape[axis]`。
    - **step**  (int，可选) - 嵌入的步长。默认为 `1`。
    - **name**  (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 `None`。

返回
:::::::::

Tensor， 与 ``x`` 数据类型与形状相同。

代码示例
:::::::::

COPY-FROM: paddle.slice_scatter
