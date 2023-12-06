.. _cn_api_paddle_full:

full
-------------------------------

.. py:function:: paddle.full(shape, fill_value, dtype=None, name=None)



创建形状大小为 ``shape`` 并且数据类型为 ``dtype``  的 Tensor，其中元素值均为 ``fill_value`` 。

参数
::::::::::::

    - **shape** (list|tuple|Tensor) – 指定创建 Tensor 的形状(shape)，数据类型为 int32 或者 int64。如果 ``shape`` 是一个 list 或 tuple ，它的每个元素应该是整数或者是形状为[]的 0-D Tensor 。如果 ``shape`` 是一个 Tensor ，它应该是一个表示 list 的 1-D 张量。
    - **fill_value** (bool|float|int|Tensor) - 用于初始化输出 Tensor 的常量数据的值。如果 fill_value 是一个 Tensor ，它应该是一个表示标量的 0-D Tensor。注意：该参数不可超过输出变量数据类型的表示范围。
    - **dtype** （np.dtype|str，可选）- 输出变量的数据类型，可以是 float16、float32、float64、int32、int64。如果 dytpe 为 None，则创建的 Tensor 的数据类型为 float32。默认值为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
返回一个存储结果的 Tensor，数据类型和 dtype 相同。


代码示例
::::::::::::

COPY-FROM: paddle.full
