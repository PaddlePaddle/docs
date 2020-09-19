.. _cn_api_tensor_cn_unique:

unique
-------------------------------

.. py:function:: paddle.unique(x, return_index=False, return_inverse=False, return_counts=False, axis=None, dtype="int64", name=None)

返回Tensor按升序排序后的独有元素。 

参数：
    - **x** (Tensor) - 输入的 `Tensor` ，数据类型为：float32、float64、int32、int64。
    - **return_index** (bool, 可选) - 如果为True，则还返回独有元素在输入Tensor中的索引。
    - **return_inverse** (bool, 可选) - 如果为True，则还返回输入Tensor的元素对应在独有元素中的索引，该索引可用于重构输入Tensor。
    - **return_counts** (bool, 可选) - 如果为True，则还返回每个独有元素在输入Tensor中的个数。
    - **axis** (int, 可选) - 指定选取独有元素的轴。默认值为None，将输入平铺为1-D的Tensor后再选取独有元素。
    - **dtype** (np.dtype|str, 可选) - 用于设置 `index`，`inverse` 或者 `counts` 的类型，应该为int32或者int64。默认：int64.
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：
    - **out** (Tensor) - 独有元素构成的Tensor，数据类型与输入一致。
    - **index** (Tensor, 可选) - 独有元素在输入Tensor中的索引，仅在 `return_index` 为True时返回。
    - **inverse** (Tensor, 可选) - 输入Tensor的元素对应在独有元素中的索引，仅在 `return_inverse` 为True时返回。
    - **counts** (Tensor, 可选) - 每个独有元素在输入Tensor中的个数，仅在 `return_counts` 为True时返回。

**代码示例**：

.. code-block:: python

    import numpy as np
    import paddle

    paddle.disable_static()
    x_data = np.array([2, 3, 3, 1, 5, 3])
    x = paddle.to_tensor(x_data)
    unique = paddle.unique(x)
    np_unique = unique.numpy() # [1 2 3 5]
    _, indices, inverse, counts = paddle.unique(x, return_index=True, return_inverse=True, return_counts=True)
    np_indices = indices.numpy() # [3 0 1 4]
    np_inverse = inverse.numpy() # [1 2 2 0 3 2]
    np_counts = counts.numpy() # [1 1 3 1]

    x_data = np.array([[2, 1, 3], [3, 0, 1], [2, 1, 3]])
    x = paddle.to_tensor(x_data)
    unique = paddle.unique(x)
    np_unique = unique.numpy() # [0 1 2 3]

    unique = paddle.unique(x, axis=0)
    np_unique = unique.numpy() 
    # [[2 1 3]
    #  [3 0 1]]
    


