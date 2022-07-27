.. _cn_api_tensor_cn_unique_consecutive:

unique_consecutive
-------------------------------

.. py:function:: paddle.unique_consecutive(x, return_inverse=False, return_counts=False, axis=None, dtype="int64", name=None)

将 Tensor 中连续重复的元素进行去重，返回连续不重复的 Tensor。

参数
::::::::::::

    - **x** (Tensor) - 输入的 `Tensor`，数据类型为：float32、float64、int32、int64。
    - **return_inverse** (bool，可选) - 如果为 True，则还返回输入 Tensor 的元素对应在连续不重复元素中的索引，该索引可用于重构输入 Tensor。默认：False。
    - **return_counts** (bool，可选) - 如果为 True，则还返回每个连续不重复元素在输入 Tensor 中的个数。默认：False。
    - **axis** (int，可选) - 指定选取连续不重复元素的轴。默认值为 None，将输入平铺为 1-D 的 Tensor 后再选取连续不重复元素。默认：None。
    - **dtype** (np.dtype|str，可选) - 用于设置 `inverse` 或者 `counts` 的类型，应该为 int32 或者 int64。默认：int64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

    - **out** (Tensor) - 连续不重复元素构成的 Tensor，数据类型与输入一致。
    - **inverse** (Tensor，可选) - 输入 Tensor 的元素对应在连续不重复元素中的索引，仅在 `return_inverse` 为 True 时返回。
    - **counts** (Tensor，可选) - 每个连续不重复元素在输入 Tensor 中的个数，仅在 `return_counts` 为 True 时返回。

代码示例
::::::::::::

COPY-FROM: paddle.unique_consecutive
