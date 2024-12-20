.. _cn_api_paddle_unfold:

unfold
--------------------------------

.. py:function:: paddle.unfold(x, axis, size, step, name=None)

返回 x 的一个 view Tensor。以滑动窗口式提取 x 的值。

仅在动态图下可用，返回的 Tensor 和 x 共享内存。

下图展示了一个元素值为 0-8 且 Shape 为 [9] 的 Tensor 如何使用 ``unfold`` 方法在第一个维度（axis = 0），使用大小为 2 的窗口（size = 2）根据 4 的步长（step = 4）进行采样。

.. image:: ../../images/api_legend/unfold.png
    :width: 700
    :alt: 图例

参数
:::::::::

    - **x** (Tensor) - 输入多维 Tensor，可选的数据类型为 'float16'、'float32'、'float64'、'int16'、'int32'、'int64'、'bool'、'uint16'。
    - **axis** (int) - 表示需要提取的维度。如果 ``axis < 0``，则维度为 ``rank(x) + axis``。
    - **size** (int) - 表示需要提取的窗口长度。
    - **step** (int) - 表示每次提取跳跃的步长。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
Tensor，x 的一个 view Tensor。


代码示例
:::::::::

COPY-FROM: paddle.unfold
