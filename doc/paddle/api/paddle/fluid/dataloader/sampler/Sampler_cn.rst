.. _cn_api_io_cn_Sampler:

Sampler
-------------------------------

.. py:class:: paddle.io.Sampler(data_source=None)

概括数据集采样器行为和方法的基类。

所有数据集采样器必须继承这个基类，并实现以下方法：

``__iter__``: 迭代返回数据样本下标

``__len__``: ``data_source`` 中的样本数

参数:
    - **data_source** (Dataset) - 此参数必须是 ``paddle.io.Dataset`` 或 ``paddle.io.IterableDataset`` 的一个子类实例或实现了 ``__len__`` 的Python对象，用于生成样本下标。默认值为None。

可见 ``paddle.io.BatchSampler`` 和 ``paddle.io.DataLoader``

返回：返回样本下标的迭代器。

返回类型: Sampler

**代码示例**

.. code-block:: python

    from paddle.io import Dataset, Sampler
    
    class RandomDataset(Dataset):
        def __init__(self, num_samples):
            self.num_samples = num_samples
    
        def __getitem__(self, idx):
            image = np.random.random([784]).astype('float32')
            label = np.random.randint(0, 9, (1, )).astype('int64')
            return image, label
        
        def __len__(self):
            return self.num_samples
    
    class MySampler(Sampler):
        def __init__(self, data_source):
            self.data_source = data_source
    
        def __iter__(self):
            return iter(range(len(self.data_source)))
    
        def __len__(self):
            return len(self.data_source)
    
    sampler = MySampler(data_source=RandomDataset(100))
    
    for index in sampler:
        print(index)

