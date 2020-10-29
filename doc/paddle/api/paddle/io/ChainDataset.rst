.. _cn_api_io_cn_ChainDataset:

ChainDataset
-------------------------------

.. py:class:: paddle.io.ChainDataset

将多个流式数据集级联的数据集。

用于级联的数据集须都是 ``paddle.io.IterableDataset`` 数据集，将各流式数据集按顺序级联为一个数据集。

**代码示例**

.. code-block:: python

    import numpy as np
    import paddle
    from paddle.io import IterableDataset, ChainDataset


    # define a random dataset
    class RandomDataset(IterableDataset):
        def __init__(self, num_samples):
            self.num_samples = num_samples

        def __iter__(self):
            for i in range(10):
                image = np.random.random([32]).astype('float32')
                label = np.random.randint(0, 9, (1, )).astype('int64')
                yield image, label

    dataset = ChainDataset([RandomDataset(10), RandomDataset(10)])
    for image, label in iter(dataset):
        print(image, label)

