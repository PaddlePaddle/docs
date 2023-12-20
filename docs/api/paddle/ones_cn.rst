.. _cn_api_paddle_ones:

ones
-------------------------------

.. py:function:: paddle.ones(shape, dtype=None, name=None)



创建一个形状为 ``shape``、数据类型为 ``dtype`` 且值全为 1 的 Tensor。

参数
:::::::::

    - **shape** (tuple|list|Tensor) - 要创建的 Tensor 的形状，``shape`` 的数据类型为 int32 或 int64。若 shape 是 list 或 tuple 类型，其元素应为整数或零维张量，若 shape 是 Tensor 类型，其元素应为表示列表的一维张量。
    - **dtype** (np.dtype|str，可选) - 要创建的 Tensor 的数据类型，可以为 bool、float16、float32、float64、int32 或 int64。如果 ``dtype`` 为 None，那么数据类型为 float32。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
Tensor，每个元素都是 1，形状为 ``shape``，数据类型为 ``dtype``。


代码示例
:::::::::

COPY-FROM: paddle.ones
