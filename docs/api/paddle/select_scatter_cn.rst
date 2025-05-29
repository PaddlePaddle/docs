.. _cn_api_paddle_select_scatter:

select_scatter
-------------------------------

.. py:function:: paddle.select_scatter(x, values, axis, index, name=None)
将 ``values`` 矩阵的值嵌入到 ``x`` 矩阵的第 ``axis`` 维的 ``index`` 列

参数
:::::::::
    - **x**  (Tensor) - 输入的 Tensor 作为目标矩阵，数据类型为： `bool`、 `float16`、 `float32`、 `float64`、 `uint8`、 `int8`、 `int16`、 `int32`、 `int64`、 `bfloat16`、 `complex64`、 `complex128`。
    - **values**  (Tensor) - 需要插入的值，形状需要与 ``x`` 矩阵除去第 ``axis`` 维后的形状一致，数据类型为： `bool`、 `float16`、 `float32`、 `float64`、 `uint8`、 `int8`、 `int16`、 `int32`、 `int64`、 `bfloat16`、 `complex64`、 `complex128`。
    - **axis**  (int) - 指定沿着哪个维度嵌入对应的值，数据类型为：int。
    - **index**  (int) - 指定沿着 ``axis`` 维的哪一列嵌入对应的值，数据类型为：int。
    - **name**  (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::

输出 Tensor， ``x`` 矩阵的第 ``axis`` 维的第 ``index`` 列会被插入 ``value``，与 ``x`` 数据类型相同。

代码示例
:::::::::

COPY-FROM: paddle.select_scatter
