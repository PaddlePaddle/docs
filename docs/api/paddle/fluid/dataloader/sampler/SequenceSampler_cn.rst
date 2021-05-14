.. _cn_api_io_cn_SequenceSampler:

SequenceSampler
-------------------------------

.. py:class:: paddle.io.SequenceSampler(data_source=None)

顺序迭代 ``data_source`` 返回样本下标，即一次返回 ``0, 1, 2, ..., len(data_source) - 1``

参数:
    - **data_source** (Dataset) - 此参数必须是 ``paddle.io.Dataset`` 或 ``paddle.io.IterableDataset`` 的一个子类实例或实现了 ``__len__`` 的Python对象，用于生成样本下标。默认值为None。

返回：返回样本下标的迭代器。

返回类型: SequenceSampler 

**代码示例**

.. code-block:: python

    from paddle.io import Dataset, SequenceSampler
    
    class RandomDataset(Dataset):
        def __init__(self, num_samples):
            self.num_samples = num_samples
    
        def __getitem__(self, idx):
            image = np.random.random([784]).astype('float32')
            label = np.random.randint(0, 9, (1, )).astype('int64')
            return image, label
        
        def __len__(self):
            return self.num_samples
    
    sampler = SequenceSampler(data_source=RandomDataset(100))
    
    for index in sampler:
        print(index)

