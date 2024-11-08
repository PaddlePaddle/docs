.. _cn_api_paddle_index_fill:

index_fill
-------------------------------

.. py:function:: paddle.index_fill(x, index, axis, value, name=None)

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

**图解说明**：

    一个形状为 [3, 3] 的二维张量，通过 index_fill 操作，当 axis = 0，索引张量 index 为 [0, 2]，填充值 value = -1 时，将第一行和第三行的所有元素填充为 -1，从而得到形状仍为 [3, 3] 但元素部分改变的新张量。

    .. figure:: ../../images/api_legend/index_fill.png
        :width: 500
        :align: center

代码示例
::::::::::::

COPY-FROM: paddle.index_fill

更多关于 outplace 操作的介绍请参考 `3.1.3 原位（Inplace）操作和非原位（Outplace）操作的区别`_ 了解详情。

.. _3.1.3 原位（Inplace）操作和非原位（Outplace）操作的区别: https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/beginner/tensor_cn.html#id3
