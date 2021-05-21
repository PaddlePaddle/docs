.. _cn_api_io_cn_Dataset:

Dataset
-------------------------------

.. py:class:: paddle.io.Dataset

概述Dataset的方法和行为的抽象类。

映射式(map-style)数据集需要继承这个基类，映射式数据集为可以通过一个键值索引并获取指定样本的数据集，所有映射式数据集须实现以下方法：

``__getitem__``: 根据给定索引获取数据集中指定样本，在 ``paddle.io.DataLoader`` 中需要使用此函数通过下标获取样本。

``__len__``: 返回数据集样本个数， ``paddle.io.BatchSampler`` 中需要样本个数生成下标序列。

见 ``paddle.io.DataLoader`` 。

**代码示例**

.. code-block:: python

    import numpy as np
    from paddle.io import Dataset

    # define a random dataset
    class RandomDataset(Dataset):
        def __init__(self, num_samples):
            self.num_samples = num_samples

        def __getitem__(self, idx):
            image = np.random.random([784]).astype('float32')
            label = np.random.randint(0, 9, (1, )).astype('int64')
            return image, label

        def __len__(self):
            return self.num_samples

    dataset = RandomDataset(10)
    for i in range(len(dataset)):
        print(dataset[i])

