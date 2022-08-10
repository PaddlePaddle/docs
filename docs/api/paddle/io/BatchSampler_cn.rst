.. _cn_api_io_cn_BatchSampler:

BatchSampler
-------------------------------

.. py:class:: paddle.io.BatchSampler(dataset=None, sampler=None, shuffle=False, batch_size=1, drop_last=False)

批采样器的基础实现，用于 ``paddle.io.DataLoader`` 中迭代式获取 mini-batch 的样本下标数组，数组长度与 ``batch_size`` 一致。

所有用于 ``paddle.io.DataLoader`` 中的批采样器都必须是 ``paddle.io.BatchSampler`` 的子类并实现以下方法：

``__iter__``：迭代式返回批样本下标数组。

``__len__``：每 epoch 中 mini-batch 数。

参数
::::::::::::

    - **dataset** (Dataset) - 此参数必须是 ``paddle.io.Dataset`` 或 ``paddle.io.IterableDataset`` 的一个子类实例或实现了 ``__len__`` 的 Python 对象，用于生成样本下标。默认值为 None。
    - **sampler** (Sampler) - 此参数必须是 ``paddle.io.Sampler`` 的子类实例，用于迭代式获取样本下标。``dataset`` 和 ``sampler`` 参数只能设置一个。默认值为 None。
    - **shuffle** (bool) - 是否需要在生成样本下标时打乱顺序。默认值为 False。
    - **batch_size** (int) - 每 mini-batch 中包含的样本数。默认值为 1。
    - **drop_last** (bool) - 是否需要丢弃最后无法凑整一个 mini-batch 的样本。默认值为 False。

见 ``paddle.io.DataLoader`` 。

返回
::::::::::::
BatchSampler，返回样本下标数组的迭代器。


代码示例
::::::::::::

COPY-FROM: paddle.io.BatchSampler
