.. _cn_api_io_SubsetDataset:

SubsetDataset
-------------------------------

.. py:class:: paddle.io.SubsetDataset

用于构造一个数据集级的数据子数据集。

给定原数据集合的指标数组，可以以此数组构造原数据集合的子数据集合。

参数:
    - **datasets** - 原数据集。
    - **indices** - 用于提取子集的原数据集合指标数组。

返回：list，原数据集合的子集列表。

**代码示例**

.. code-block:: python

    import paddle
    from paddle.io import Subset
    # example 1:
    a = paddle.io.Subset(dataset=range(1, 4), indices=[0, 2])
    print(list(a))
    # [1, 3]
    # example 2:
    b = paddle.io.Subset(dataset=range(1, 4), indices=[1, 1])
    print(list(b))
    # [2, 2]

