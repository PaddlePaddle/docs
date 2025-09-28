.. _cn_api_paddle_io_random_split:

random_split
-------------------------------

.. py:class:: paddle.io.random_split(dataset, lengths, generator=None)

给定子集合 dataset 的长度数组，随机切分出原数据集合的非重复子集合。

参数
::::::::::::

    - **dataset** (Dataset) - 此参数必须是 ``paddle.io.Dataset`` 或 ``paddle.io.IterableDataset`` 的一个子类实例或实现了 ``__len__`` 的 Python 对象，用于生成样本下标。默认值为 None。
    - **lengths** (list) - 总和为原数组长度，表示子集合长度数组；或总和为 1.0，表示子集合长度占比的数组。
    - **generator** (Generator，可选) - 指定采样 ``data_source`` 的采样器。默认值为 None。

返回
::::::::::::
 list，返回按给定长度数组描述随机分割的原数据集合的非重复子集合。


代码示例
::::::::::::

COPY-FROM: paddle.io.random_split
