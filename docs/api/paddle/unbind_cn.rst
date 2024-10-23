.. _cn_api_paddle_unbind:

unbind
-------------------------------

.. py:function:: paddle.unbind(input, axis=0)

下图展示了一个形状为 [2, 3, 4] 的三维张量在通过沿 axis0 进行 unbind 操作之后转变为2个形状为 [3, 4] 的二维张量。值得注意的是沿着 axis0 进行 unbind 操作仅能返回2个张量，沿着 axis1 进行 unbind 进行操作仅能返回3个张量。

.. image:: ../../images/api_legend/unbind.png
   :width: 700
   :alt: 图例

参数
:::::::::
       - **input** (Tensor) - 输入变量，数据类型为 float16、loat32、float64、int32、int64、complex64、complex128 的多维 Tensor。
       - **axis** (int32|int64，可选) - 数据类型为 int32 或 int64，表示需要分割的维度。如果 axis < 0，则划分的维度为 rank(input) + axis。默认值为 0。

返回
:::::::::
Tensor，分割后的 Tensor 列表。

代码示例
:::::::::

COPY-FROM: paddle.unbind
