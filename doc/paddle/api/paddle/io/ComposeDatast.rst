.. _cn_api_io_ComposeDataset:

ComposeDataset
-------------------------------

.. py:class:: paddle.io.ComposeDataset

由多个数据集的字段组成的数据集。

这个数据集用于将多个映射式(map-style)且长度相等数据集按字段组合为一个新的数据集。

参数:
    - **datasets** (list of Dataset) - 待组合的多个数据集。

返回：Dataset，字段组合后的数据集

**代码示例**

.. code-block:: python

    import numpy as np
    import paddle
    from paddle.io import Dataset, ComposeDataset


    # define a random dataset
    class RandomDataset(Dataset):
        def __init__(self, num_samples):
            self.num_samples = num_samples

        def __getitem__(self, idx):
            image = np.random.random([32]).astype('float32')
            label = np.random.randint(0, 9, (1, )).astype('int64')
            return image, label

        def __len__(self):
            return self.num_samples

    dataset = ComposeDataset([RandomDataset(10), RandomDataset(10)])
    for i in range(len(dataset)):
        image1, label1, image2, label2 = dataset[i]
        print(image1)
        print(label1)
        print(image2)
        print(label2)

