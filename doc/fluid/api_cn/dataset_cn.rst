#################
 fluid.dataset
#################






.. _cn_api_fluid_dataset_DatasetFactory:

DatasetFactory
-------------------------------

.. py:class:: paddle.fluid.dataset.DatasetFactory

DatasetFactory是一个按数据集名称创建数据集的 "工厂" ，可以创建“QueueDataset”或“InMemoryDataset”，默认为“QueueDataset”。

**代码示例**

.. code-block:: python

    dataset = paddle.fluid.DatasetFactory.create_dataset(“InMemoryDataset”)

.. py:method:: create_dataset(datafeed_class='QueueDataset')

创建“QueueDataset”或“InMemoryDataset”，默认为“QueueDataset”。














