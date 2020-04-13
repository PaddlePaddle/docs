.. _cn_api_fluid_io_BatchSampler:

BatchSampler
-------------------------------

.. py:class:: paddle.fluid.io.BatchSampler(dataset=None, indices=None, shuffle=False, batch_size=1, drop_last=False)

``fluid.io.DataLoader`` 使用的批次索引采样器，其可以迭代返回mini-batch的索引列表(长度为 ``batch_size`` ，内容为样本索引)。
``fluid.io.DataLoader`` 的 ``batch_sampler`` 参数必须为 ``BatchSampler`` 及其子类实例。 ``BatchSampler`` 子类须实现如下两个方法：

``__iter__`` : 迭代返回mini-batch索引列表。

``__len__`` : 每个epoch中的mini-batch个数。

参数:
    - **dataset** (Dataset) - ``fluid.io.Dataset`` 实例或者实现了 ``__len__`` 接口的python对象，用于生成 ``dataset`` 长度范围的索引。默认值为None。
    - **indices** (list|tuple) - 用于迭代的下标，``dataset`` 的替代参数， ``dataset`` 和 ``indices`` 必须设置其中之一。默认值为None。
    - **shuffle** (bool) - 迭代返回索引之前是否对索引打乱顺序。默认值为False。
    - **batch_size** (int) - 每mini-batch中的索引下标个数。默认值为1。
    - **drop_last** (int) - 是否丢弃因数据集样本数不能被 ``batch_size`` 整除而产生的最后一个不完整的mini-batch索引。默认值为False。

返回：迭代索引列表的迭代器

返回类型: BatchSampler

**代码示例**

.. code-block:: python

    from paddle.fluid.io import BatchSampler, MNIST

    # init with indices
    bs = BatchSampler(indices=list(range(1000)),
                      shuffle=True,
                      batch_size=8,
                      drop_last=True)

    for batch_indices in bs:
        print(batch_indices)

    # init with dataset
    bs = BatchSampler(dataset=MNIST(mode='test')),
                      shuffle=False,
                      batch_size=16,
                      drop_last=False)

    for batch_indices in bs:
        print(batch_indices)
