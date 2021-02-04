.. _cn_api_io_cn_BatchSampler:

BatchSampler
-------------------------------

.. py:class:: paddle.io.BatchSampler(dataset=None, sampler=None, shuffle=Fasle, batch_size=1, drop_last=False)

批采样器的基础实现，用于 ``paddle.io.DataLoader`` 中迭代式获取mini-batch的样本下标数组，数组长度与 ``batch_size`` 一致。

所有用于 ``paddle.io.DataLoader`` 中的批采样器都必须是 ``paddle.io.BatchSampler`` 的子类并实现以下方法:

``__iter__``: 迭代式返回批样本下标数组。

``__len__``: 每epoch中mini-batch数。

参数:
    - **dataset** (Dataset) - 此参数必须是 ``paddle.io.Dataset`` 或 ``paddle.io.IterableDataset`` 的一个子类实例或实现了 ``__len__`` 的Python对象，用于生成样本下标。默认值为None。
    - **sampler** (Sampler) - 此参数必须是 ``paddle.io.Sampler`` 的子类实例，用于迭代式获取样本下标。``dataset`` 和 ``sampler`` 参数只能设置一个。默认值为None。
    - **shuffle** (bool) - 是否需要在生成样本下标时打乱顺序。默认值为False。
    - **batch_size** (int) - 每mini-batch中包含的样本数。默认值为1。
    - **drop_last** (bool) - 是否需要丢弃最后无法凑整一个mini-batch的样本。默认值为False。

见 ``paddle.io.DataLoader`` 。

返回：返回样本下标数组的迭代器。

返回类型: BatchSampler

**代码示例**

.. code-block:: python

    from paddle.io import RandomSampler, BatchSampler, Dataset

    # init with dataset
    class RandomDataset(Dataset):
        def __init__(self, num_samples):
            self.num_samples = num_samples

        def __getitem__(self, idx):
            image = np.random.random([784]).astype('float32')
            label = np.random.randint(0, 9, (1, )).astype('int64')
            return image, label

        def __len__(self):
            return self.num_samples

    bs = BatchSampler(dataset=RandomDataset(100),
                  shuffle=False,
                  batch_size=16,
                  drop_last=False)

    for batch_indices in bs:
        print(batch_indices)

    # init with sampler
    sampler = RandomSampler(RandomDataset(100))
    bs = BatchSampler(sampler=sampler,
                  batch_size=8,
                  drop_last=True)

    for batch_indices in bs:
        print(batch_indices)
