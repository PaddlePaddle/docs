.. _cn_api_io_cn_IterableDataset:

IterableDataset
-------------------------------

.. py:class:: paddle.io.IterableDataset

概述迭代式数据集的方法和行为的抽象类。

迭代式(iterable style)数据集需要继承这个基类，迭代式数据集为只能依次迭代式获取样本的数据集，类似 Python 中的迭代器，所有迭代式数据集须实现以下方法：

``__iter__``：依次返回数据赝本。

.. note::
    迭代式数据集不需要实现 ``__getitem__`` 和 ``__len__``，也不可以调用迭代式数据集的这两个方法。

见 ``paddle.io.DataLoader`` 。

代码示例 1
::::::::::::

.. code-block:: python

    import numpy as np
    from paddle.io import IterableDataset

    # define a random dataset
    class RandomDataset(IterableDataset):
        def __init__(self, num_samples):
            self.num_samples = num_samples

        def __iter__(self):
            for i in range(self.num_samples):
                image = np.random.random([784]).astype('float32')
                label = np.random.randint(0, 9, (1, )).astype('int64')
                yield image, label

    dataset = RandomDataset(10)
    for img, lbl in dataset:
        print(img, lbl)

当 ``paddle.io.DataLoader`` 中 ``num_workers > 0`` 时，每个子进程都会遍历全量的数据集返回全量样本，所以数据集会重复 ``num_workers``
次，如果需要数据集样本不会重复返回，可通过如下两种方法避免样本重复，两种方法中都需要通过 ``paddle.io.get_worker_info`` 获取各子进程的信息。


代码示例 2
::::::::::::

通过 ``__iter__`` 函数划分各子进程的数据

.. code-block:: python

    import math
    import paddle
    import numpy as np
    from paddle.io import IterableDataset, DataLoader, get_worker_info

    class SplitedIterableDataset(IterableDataset):
        def __init__(self, start, end):
            self.start = start
            self.end = end

        def __iter__(self):
            worker_info = get_worker_info()
            if worker_info is None:
                iter_start = self.start
                iter_end = self.end
            else:
                per_worker = int(
                    math.ceil((self.end - self.start) / float(
                        worker_info.num_workers)))
                worker_id = worker_info.id
                iter_start = self.start + worker_id * per_worker
                iter_end = min(iter_start + per_worker, self.end)

            for i in range(iter_start, iter_end):
                yield np.array([i])

    dataset = SplitedIterableDataset(start=2, end=9)
    dataloader = DataLoader(
        dataset,
        num_workers=2,
        batch_size=1,
        drop_last=True)

    for data in dataloader:
        print(data)
    # outputs: [2, 5, 3, 6, 4, 7]


代码示例 3
::::::::::::

通过各子进程初始化函数 ``worker_inif_fn`` 划分子进程数据

.. code-block:: python

    import math
    import paddle
    import numpy as np
    from paddle.io import IterableDataset, DataLoader, get_worker_info

    class RangeIterableDataset(IterableDataset):
        def __init__(self, start, end):
            self.start = start
            self.end = end

        def __iter__(self):
            for i in range(self.start, self.end):
                yield np.array([i])

    dataset = RangeIterableDataset(start=2, end=9)

    def worker_init_fn(worker_id):
        worker_info = get_worker_info()

        dataset = worker_info.dataset
        start = dataset.start
        end = dataset.end
        num_per_worker = int(
            math.ceil((end - start) / float(worker_info.num_workers)))

        worker_id = worker_info.id
        dataset.start = start + worker_id * num_per_worker
        dataset.end = min(dataset.start + num_per_worker, end)

    dataloader = DataLoader(
        dataset,
        num_workers=2,
        batch_size=1,
        drop_last=True,
        worker_init_fn=worker_init_fn)

    for data in dataloader:
        print(data)
    # outputs: [2, 5, 3, 6, 4, 7]
