.. _cn_api_io_cn_SequenceSampler:

SequenceSampler
-------------------------------

.. py:class:: paddle.io.SequenceSampler(data_source=None)

顺序迭代 ``data_source`` 返回样本下标，即一次返回 ``0, 1, 2, ..., len(data_source) - 1``

参数
::::::::::::

    - **data_source** (Dataset) - 此参数必须是 ``paddle.io.Dataset`` 或 ``paddle.io.IterableDataset`` 的一个子类实例或实现了 ``__len__`` 的 Python 对象，用于生成样本下标。默认值为 None。

返回
::::::::::::
SequenceSampler，返回样本下标的迭代器。


代码示例
::::::::::::

COPY-FROM: paddle.io.SequenceSampler
