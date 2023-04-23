.. _cn_api_io_cn_Sampler:

Sampler
-------------------------------

.. py:class:: paddle.io.Sampler(data_source=None)

概括数据集采样器行为和方法的基类。

所有数据集采样器必须继承这个基类，并实现以下方法：

``__iter__``：迭代返回数据样本下标

``__len__``: ``data_source`` 中的样本数

参数
::::::::::::

    - **data_source** (Dataset) - 此参数必须是 ``paddle.io.Dataset`` 或 ``paddle.io.IterableDataset`` 的一个子类实例或实现了 ``__len__`` 的 Python 对象，用于生成样本下标。默认值为 None。

可见 ``paddle.io.BatchSampler`` 和 ``paddle.io.DataLoader``

返回
::::::::::::
Sampler，返回样本下标的迭代器。


代码示例
::::::::::::

COPY-FROM: paddle.io.Sampler
