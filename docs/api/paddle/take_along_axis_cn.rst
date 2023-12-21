.. _cn_api_paddle_take_along_axis:

take_along_axis
-------------------------------

.. py:function:: paddle.take_along_axis(arr, indices, axis, broadcast=True)
基于输入索引矩阵，沿着指定 axis 从 arr 矩阵里选取 1d 切片。索引矩阵必须和 arr 矩阵有相同的维度，需要能够 broadcast 与 arr 矩阵对齐。

参数
:::::::::

- **arr**  (Tensor) - 输入的 Tensor 作为源矩阵，数据类型为：float32、float64。
- **indices**  (Tensor) - 索引矩阵，包含沿轴提取 1d 切片的下标，必须和 arr 矩阵有相同的维度，需要能够 broadcast 与 arr 矩阵对齐，数据类型为：int、int64。
- **axis**  (int) - 指定沿着哪个维度获取对应的值，数据类型为：int。
- **broadcast** (bool，可选) - 是否广播 ``index`` 矩阵，默认为 ``True``。

返回
:::::::::

输出 Tensor，包含 indeces 矩阵选定的元素，与 ``arr`` 数据类型相同。

代码示例
:::::::::


COPY-FROM: paddle.take_along_axis
