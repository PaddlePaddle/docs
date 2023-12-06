.. _cn_api_paddle_index_fill_:

index_fill\_
-------------------------------

.. py:function:: paddle.index_fill_(x, index, axis, value, name=None)

依据指定的轴 ``axis`` 和索引 ``indices`` 将指定位置的 ``x`` 填充为 ``value`` 。

参数
:::::::::

    - **x** （Tensor）– 输入 Tensor。 ``x`` 的数据类型可以是 float16, float32，float64，int32，int64。
    - **index** （Tensor）– 包含索引下标的 1-D Tensor。数据类型可以是 int32，int64。
    - **axis**    (int) – 索引轴。数据类型为 int。
    - **value** （float）– 用于填充目标张量的值。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::

Tensor，返回一个数据类型同输入的 Tensor。


代码示例
::::::::::::

COPY-FROM: paddle.index_fill_
