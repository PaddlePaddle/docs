.. _cn_api_paddle_histc:

histc
-------------------------------

.. py:function:: paddle.histc(input, bins=100, min=0.0, max=0.0, name=None, *, out=None)

计算 Tensor 的直方图。

元素被分配到 min 和 max 之间的等宽区间中（包含边界）。如果 min 和 max 都为零，则使用数据的最小值和最大值。

参数
:::::::::
    - **input** (Tensor) - 输入 Tensor。
    - **bins** (int，可选) - 直方图区间数，默认值为 100。
    - **min** (float，可选) - 范围的下端（包含），默认值为 0.0。
    - **max** (float，可选) - 范围的上端（包含），默认值为 0.0。
    - **name** (str，可选) - 操作名称，默认值为 None。

关键字参数
:::::::::
    - **out** (Tensor，可选) - 输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 中，默认值为 ``None``。

返回
:::::::::
Tensor：直方图 Tensor，数据类型为 float32。

代码示例
:::::::::

COPY-FROM: paddle.histc
