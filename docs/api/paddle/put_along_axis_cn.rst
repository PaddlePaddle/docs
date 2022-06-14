.. _cn_api_paddle_tensor_put_along_axis:

put_along_axis
-------------------------------

.. py:function:: paddle.put_along_axis(arr, indices, values, axis, reduce='assign')
基于输入index矩阵，将输入value沿着指定axis放置入arr矩阵。索引矩阵和value必须和arr矩阵有相同的维度，需要能够broadcast与arr矩阵对齐。

参数
:::::::::

- **arr**  (Tensor) - 输入的Tensor 作为目标矩阵，数据类型为：float32、float64。
- **indices**  (Tensor) - 索引矩阵，包含沿轴提取1d切片的下标，必须和arr矩阵有相同的维度，需要能够broadcast与arr矩阵对齐，数据类型为: int、int64。
- **value** （float）- 需要插入的值，形状和维度需要能够被broadcast与indices矩阵匹配，数据类型为: float32、float64。
- **axis**  (int) - 指定沿着哪个维度获取对应的值，数据类型为: int。
- **reduce** (str，可选) - 归约操作类型，默认为 ``assign`` ，可选为 ``add`` 或 ``multiple``.不同的规约操作插入值value对于输入矩阵arr会有不同的行为，如为 ``assgin`` 则覆盖输入矩阵，``add`` 则累加至输入矩阵，``multiple`` 则累乘至输入矩阵。

返回
:::::::::

- **out** (Tensor) - 输出Tensor，indeces矩阵选定的下标会被插入value，与 ``arr`` 数据类型相同。

代码示例
:::::::::


COPY-FROM: paddle.put_along_axis:code-example1

