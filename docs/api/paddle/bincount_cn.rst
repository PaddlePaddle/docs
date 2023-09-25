.. _cn_api_paddle_bincount:

bincount
-------------------------------

.. py:function:: paddle.bincount(x, weights=None, minlength=0, name=None)

统计输入 Tensor 中每个元素出现的次数，如果传入 weights Tensor 则每次计数加一时会乘以 weights Tensor 对应的值。

参数
::::::::::::

    - **x** (Tensor) - 输入 Tensor。必须是一维 Tensor，其中元素必须大于等于 0，数据类型为 int32, int64。
    - **weights** (Tensor，可选) - weights Tensor，代表输入 Tensor 中每个元素的权重。长度必须与输入 Tensor 相同。数据类型为 int32, int64, float32 或 float64。默认为 None
    - **minlength** (int，可选) - 输出 Tensor 的最小长度，如果大于输入 Tensor 中的最大值，则多出的位置补 0。该值必须大于等于 0。默认为 0。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，维度为 1。

代码示例
::::::::::::

COPY-FROM: paddle.bincount
