.. _cn_api_io_cn_DistributedBatchSampler:

DistributedBatchSampler
-------------------------------

.. py:class:: paddle.io.DistributedBatchSampler(dataset=None, batch_size, num_replicas=None, rank=None, shuffle=False, drop_last=False)

分布式批采样器加载数据的一个子集。每个进程可以传递给DataLoader一个DistributedBatchSampler的实例，每个进程加载原始数据的一个子集。


.. note::
  假定Dataset的大小是固定的。

参数:
    - **dataset** (paddle.io.Dataset) - 此参数必须是 ``paddle.io.Dataset`` 的一个子类实例或实现了 ``__len__`` 的Python对象，用于生成样本下标。默认值为None。
    - **batch_size** (int) - 每mini-batch中包含的样本数。
    - **num_replicas** (int, optional) - 分布式训练时的进程个数。如果是None，会依据 ``paddle.distributed.ParallenEnv`` 获取值。默认是None。
    - **rank** (int, optional) - num_replicas个进程中的进程序号。如果是None，会依据 ``paddle.distributed.ParallenEnv`` 获取值。默认是None。
    - **shuffle** (bool) - 是否需要在生成样本下标时打乱顺序。默认值为False。
    - **drop_last** (bool) - 是否需要丢弃最后无法凑整一个mini-batch的样本。默认值为False。


返回：返回样本下标数组的迭代器。

返回类型: DistributedBatchSampler

**代码示例**

.. code-block:: python

    import numpy as np

    from paddle.io import Dataset, DistributedBatchSampler

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
  
    dataset = RandomDataset(100)
    sampler = DistributedBatchSampler(dataset, batch_size=64)

    for data in sampler:
        # do something
        break

.. py:function:: set_epoch(epoch)

设置epoch数。当设置``shuffle=True``时，此epoch被用作随机种子。默认情况下，用户可以不用此接口设置，每个epoch时，所有的进程(workers)使用不同的顺序。如果每个epoch设置相同的数字，每个epoch数据的读取顺序将会相同。

参数：
    - **epoch** (int) - epoch数。

**代码示例**

.. code-block:: python

    import numpy as np
    
    from paddle.io import Dataset, DistributedBatchSampler
    
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
    
    dataset = RandomDataset(100)
    sampler = DistributedBatchSampler(dataset, batch_size=64)
    
    for epoch in range(10):
        sampler.set_epoch(epoch)
