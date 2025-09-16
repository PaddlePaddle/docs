.. _cn_api_paddle_narrow:

narrow
-------------------------------

.. py:function:: paddle.narrow(input, dim, start, length)

返回输入张量 input 在指定维度 dim 上的窄切片。
该操作保留所有维度，仅沿 dim 选取索引区间 [start, start + length) 内的元素，则返回结果与输入共享内存（视图）。

参数
:::::::::
    - **input** (Tensor) - 输入张量，支持任意数据类型与形状。
    - **dim** (int) - 要窄化的维度，支持负索引。
    - **start** (int|Tensor) - 起始索引，可为 Python 整数或 0-D 的 int32/int64 张量，负值表示从维度末尾倒数。
    - **length** (int) - 从 start 开始选取的元素个数，必须 ≥ 0。

返回
:::::::::
 ``Tensor`` ，与 input 数据类型相同，形状仅在 dim 维变为 length，其余维不变。

代码示例
:::::::::

COPY-FROM: paddle.narrow
