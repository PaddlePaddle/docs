.. _cn_api_paddle_put_along_axis:

put_along_axis
-------------------------------

.. py:function:: paddle.put_along_axis(arr, indices, values, axis, reduce='assign', include_self=True, broadcast=True)
基于输入 index 矩阵，将输入 value 沿着指定 axis 放置入 arr 矩阵。索引矩阵和 value 必须和 arr 矩阵有相同的维度，需要能够 broadcast 与 arr 矩阵对齐。

参数
:::::::::

    - **arr**  (Tensor) - 输入的 Tensor 作为目标矩阵，数据类型为：float32、float64。
    - **indices**  (Tensor) - 索引矩阵，包含沿轴提取 1d 切片的下标，必须和 arr 矩阵有相同的维度，需要能够 broadcast 与 arr 矩阵对齐，数据类型为：int、int64。
    - **values** （float）- 需要插入的值，形状和维度需要能够被 broadcast 与 indices 矩阵匹配，数据类型为：float32、float64。
    - **axis**  (int) - 指定沿着哪个维度获取对应的值，数据类型为：int。
    - **reduce** (str，可选) - 归约操作类型，默认为 ``assign``，可选为 ``add``、 ``multiple``、 ``mean``、 ``amin``、 ``amax``。不同的规约操作插入值 value 对于输入矩阵 arr 会有不同的行为，如为 ``assgin`` 则覆盖输入矩阵， ``add`` 则累加至输入矩阵， ``mean`` 则计算累计平均值至输入矩阵， ``multiple`` 则累乘至输入矩阵， ``amin`` 则计算累计最小值至输入矩阵， ``amax`` 则计算累计最大值至输入矩阵。
    - **include_self** (bool，可选) - 规约时是否包含 arr 的元素，默认为 ``True``。
    - **broadcast** (bool，可选) - 是否广播 ``index`` 矩阵，默认为 ``True``。

返回
:::::::::

输出 Tensor，indeces 矩阵选定的下标会被插入 value，与 ``arr`` 数据类型相同。

代码示例
:::::::::

COPY-FROM: paddle.put_along_axis
