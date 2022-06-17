.. _cn_api_tensor_bincount:

bincount
-------------------------------

.. py:function:: paddle.bincount(x, weights=None, minlength=0, name=None):

统计输入张量中每个元素出现的次数，如果传入weights张量则每次计数加一时会乘以weights张量对应的值

参数
::::::::::::

    - **x** (Tensor) - 输入Tensor。必须是一维Tensor，其中元素必须大于等于0，数据类型为int32, int64。
    - **weights** (Tensor，可选) - weights Tensor，代表输入Tensor中每个元素的权重。长度必须与输入Tensor相同。数据类型为int32, int64, float32或float64。默认为None
    - **minlength** (int，可选) - 输出Tensor的最小长度，如果大于输入Tensor中的最大值，则多出的位置补0。该值必须大于等于0。默认为0。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，维度为1。

代码示例：
::::::::::::

COPY-FROM: paddle.bincount