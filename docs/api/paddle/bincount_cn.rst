.. _cn_api_tensor_bincount:

bincount
-------------------------------

.. py:function:: paddle.bincount(x, weights=None, minlength=0):

统计输入张量中每个元素出现的次数，如果传入weights张量则每次计数加一时会乘以weights张量对应的值

参数：
    - **x** (Tensor) - 输入Tensor。必须是一维Tensor，其中元素必须大于等于0，数据类型为int32, int64。
    - **weights** (Tensor, 可选) - weights Tensor，代表输入Tensor中每个元素的权重。长度必须与输入Tensor相同。数据类型为int32, int64, float32或float64。
    - **minlength** (int) - 输出Tensor的最小长度，如果大于输入Tensor的长度，则多出的位置补0。该值必须大于等于0。默认为0。

返回：Tensor，维度为1。

抛出异常：
    - ``ValueError`` - 当输入Tensor或weights Tensor的值或形状不合法时。
    - ``IndexError`` - 当输入minlength的值不合法时。

**代码示例**：

.. code-block:: python

    import paddle

    x = paddle.to_tensor([1, 2, 1, 4, 5])
    result1 = paddle.bincount(x)
    print(result1) # [0, 2, 1, 0, 1, 1]
    
    w = paddle.to_tensor([2.1, 0.4, 0.1, 0.5, 0.5])
    result2 = paddle.bincount(x, weights=w)
    print(result2) # [0., 2.19999981, 0.40000001, 0., 0.50000000, 0.50000000]


