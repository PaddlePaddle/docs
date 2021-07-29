.. _cn_api_tensor_cn_unique_consecutive:

unique_consecutive
-------------------------------

.. py:function:: paddle.unique_consecutive(x, return_inverse=False, return_counts=False, axis=None, dtype="int64", name=None)

将Tensor中连续重复的元素进行去重，返回连续不重复的Tensor。 

参数：
    - **x** (Tensor) - 输入的 `Tensor` ，数据类型为：float32、float64、int32、int64。
    - **return_inverse** (bool, 可选) - 如果为True，则还返回输入Tensor的元素对应在连续不重复元素中的索引，该索引可用于重构输入Tensor。默认：False.
    - **return_counts** (bool, 可选) - 如果为True，则还返回每个连续不重复元素在输入Tensor中的个数。默认：False.
    - **axis** (int, 可选) - 指定选取连续不重复元素的轴。默认值为None，将输入平铺为1-D的Tensor后再选取连续不重复元素。
    - **dtype** (np.dtype|str, 可选) - 用于设置 `inverse` 或者 `counts` 的类型，应该为int32或者int64。默认：int64.
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：
    - **out** (Tensor) - 连续不重复元素构成的Tensor，数据类型与输入一致。
    - **inverse** (Tensor, 可选) - 输入Tensor的元素对应在连续不重复元素中的索引，仅在 `return_inverse` 为True时返回。
    - **counts** (Tensor, 可选) - 每个连续不重复元素在输入Tensor中的个数，仅在 `return_counts` 为True时返回。

**代码示例**：

.. code-block:: python

    import paddle 

    x = paddle.to_tensor([1, 1, 2, 2, 3, 1, 1, 2])
    output = paddle.unique_consecutive(x) # 
    np_output = output.numpy() # [1 2 3 1 2]
    _, inverse, counts = paddle.unique_consecutive(x, return_inverse=True, return_counts=True)
    np_inverse = inverse.numpy() # [0 0 1 1 2 3 3 4]
    np_counts = inverse.numpy() # [2 2 1 2 1]

    x = paddle.to_tensor([[2, 1, 3], [3, 0, 1], [2, 1, 3], [2, 1, 3]])
    output = paddle.unique_consecutive(x, axis=0) # 
    np_output = output.numpy() # [2 1 3 0 1 2 1 3 2 1 3]

    x = paddle.to_tensor([[2, 1, 3], [3, 0, 1], [2, 1, 3], [2, 1, 3]])
    output = paddle.unique_consecutive(x, axis=0) # 
    np_output = output.numpy()
    # [[2 1 3]
    #  [3 0 1]
    #  [2 1 3]]
