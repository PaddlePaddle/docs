.. _cn_api_io_cn_RandomSampler:

RandomSampler
-------------------------------

.. py:class:: paddle.io.RandomSampler(data_source=None, replacement=False, num_samples=None, generator=None)

顺序迭代 ``data_source`` 返回样本下标，即一次返回 ``0, 1, 2, ..., len(data_source) - 1``

参数:
    - **data_source** (Dataset) - 此参数必须是 ``paddle.io.Dataset`` 或 ``paddle.io.IterableDataset`` 的一个子类实例或实现了 ``__len__`` 的Python对象，用于生成样本下标。默认值为None。
    - **replacement** (bool) - 如果为 ``False`` 则会采样整个数据集，如果为 ``True`` 则会按 ``num_samples`` 指定的样本数采集。默认值为 ``False`` 。
    - **num_samples** (int) - 如果 ``replacement`` 设置为 ``True`` 则按此参数采集对应的样本数。默认值为None。
    - **generator** (Generator) - 指定采样 ``data_source`` 的采样器。默认值为None。

返回: 返回随机采样下标的采样器

返回类型: RandomSampler 

**代码示例**

.. code-block:: python

    from paddle.io import Dataset, RandomSampler
    
    class RandomDataset(Dataset):
        def __init__(self, num_samples):
            self.num_samples = num_samples
    
        def __getitem__(self, idx):
            image = np.random.random([784]).astype('float32')
            label = np.random.randint(0, 9, (1, )).astype('int64')
            return image, label
        
        def __len__(self):
            return self.num_samples
    
    sampler = RandomSampler(data_souce=RandomDataset(100))
    
    for index in sampler:
        print(index)
