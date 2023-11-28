.. _cn_api_paddle_dsplit:

dsplit
-------------------------------

.. py:function:: paddle.dsplit(x, num_or_sections, name=None)



将输入 Tensor 沿着深度轴分割成多个子 Tensor，等价于将 paddle.split API 的参数 axis 固定为 2。

参数
:::::::::
       - **x** (Tensor) - 输入变量，数据类型为 bool、bfloat16、float16、float32、float64、uint8、int8、int32、int64 的多维 Tensor，其维度必须大于 2。
       - **num_or_sections** (int|list|tuple) - 如果 ``num_or_sections`` 是一个整数，则表示 Tensor 平均划分为相同大小子 Tensor 的数量。如果 ``num_or_sections`` 是一个 list 或 tuple，那么它的长度代表子 Tensor 的数量，它的元素可以是整数或者形状为[]的 0-D Tensor，依次代表子 Tensor 需要分割成的维度的大小。list 或 tuple 的长度不能超过输入 Tensor 轴 2 的大小。在 list 或 tuple 中，至多有一个元素值为-1，表示该值是由 ``x`` 的维度和其他 ``num_or_sections`` 中元素推断出来的。
       - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::

list[Tensor]，分割后的 Tensor 列表。


代码示例
:::::::::

COPY-FROM: paddle.dsplit
