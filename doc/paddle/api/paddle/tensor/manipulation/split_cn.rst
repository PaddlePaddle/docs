.. _cn_api_paddle_tensor_split
split
-------------------------------

.. py:function:: paddle.tensor.split(x, num_or_sections, axis=0, name=None)



该OP将输入Tensor分割成多个子Tensor。

**参数**：
       - **x** (Tensor) - 输入变量，数据类型为bool, float16, float32，float64，int32，int64的多维Tensor。
       - **num_or_sections** (int|list|tuple) - 如果 ``num_or_sections`` 是一个整数，则表示Tensor平均划分为相同大小子Tensor的数量。如果 ``num_or_sections`` 是一个list或tuple，那么它的长度代表子Tensor的数量，它的元素可以是整数或者形状为[1]的Tensor，依次代表子Tensor需要分割成的维度的大小。list或tuple的长度不能超过输入Tensor待分割的维度的大小。在list或tuple中，至多有一个元素值为-1，表示该值是由 ``x`` 的维度和其他 ``num_or_sections`` 中元素推断出来的。例如对一个维度为[4,6,6]Tensor的第三维进行分割时，指定 ``num_or_sections=[2,-1,1]`` ，输出的三个Tensor维度分别为：[4,6,2]，[4,6,3]，[4,6,1]。
       - **axis** (int|Tensor，可选) - 整数或者形状为[1]的Tensor，数据类型为int32或int64。表示需要分割的维度。如果 ``axis < 0`` ，则划分的维度为 ``rank(x) + axis`` 。默认值为0。
       - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：分割后的Tensor列表。


**代码示例**：

.. code-block:: python

    import numpy as np
    import paddle
    
    paddle.disable_static()
    # x is a Tensor which shape is [3, 9, 5]
    x_np = np.random.random([3, 9, 5]).astype("int32")
    x = paddle.to_tensor(x_np)

    out0, out1, out22 = paddle.split(x, num_or_sections=3, axis=1)
    # out0.shape [3, 3, 5]
    # out1.shape [3, 3, 5]
    # out2.shape [3, 3, 5]

    out0, out1, out2 = paddle.split(x, num_or_sections=[2, 3, 4], axis=1)
    # out0.shape [3, 2, 5]
    # out1.shape [3, 3, 5]
    # out2.shape [3, 4, 5]

    out0, out1, out2 = paddle.split(x, num_or_sections=[2, 3, -1], axis=1)
    # out0.shape [3, 2, 5]
    # out1.shape [3, 3, 5]
    # out2.shape [3, 4, 5]
    
    # axis is negative, the real axis is (rank(x) + axis) which real
    # value is 1.
    out0, out1, out2 = paddle.split(x, num_or_sections=3, axis=-2)
    # out0.shape [3, 3, 5]
    # out1.shape [3, 3, 5]
    # out2.shape [3, 3, 5]
