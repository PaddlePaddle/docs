.. _cn_api_paddle_tensor_unbind:

unbind
-------------------------------

.. py:function:: paddle.unbind(input, axis=0)




将输入 Tensor 按照指定的维度分割成多个子 Tensor。

参数
:::::::::
       - **input** (Tensor) - 输入变量，数据类型为 float16、loat32、float64、int32、int64 的多维 Tensor。
       - **axis** (int32|int64，可选) - 数据类型为 int32 或 int64，表示需要分割的维度。如果 axis < 0，则划分的维度为 rank(input) + axis。默认值为 0。

返回
:::::::::
Tensor，分割后的 Tensor 列表。

代码示例
:::::::::

COPY-FROM: paddle.unbind
