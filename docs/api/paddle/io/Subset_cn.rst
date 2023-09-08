.. _cn_api_paddle_io_Subset:

Subset
-------------------------------

.. py:class:: paddle.io.Subset(dataset, indices)
用于构造一个数据集级的数据子数据集。

给定原数据集合的指标数组，可以以此数组构造原数据集合的子数据集合。

参数
:::::::::

    - **datasets** (Dataset) - 原数据集。
    - **indices** (sequence) - 用于提取子集的原数据集合指标数组。

返回
:::::::::

list[Dataset]，原数据集合的子集列表。

代码示例
:::::::::

COPY-FROM: paddle.io.Subset
