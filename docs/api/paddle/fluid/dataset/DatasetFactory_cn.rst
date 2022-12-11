.. _cn_api_fluid_dataset_DatasetFactory:

DatasetFactory
-------------------------------

.. py:class:: paddle.fluid.dataset.DatasetFactory




DatasetFactory是一个按数据集名称创建数据集的 "工厂"，可以创建“QueueDataset”，“InMemoryDataset”或“FileInstantDataset”，默认为“QueueDataset”。


代码示例
::::::::::::

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")

方法
::::::::::::
create_dataset(datafeed_class='QueueDataset')
'''''''''

创建“QueueDataset”，“InMemoryDataset” 或 “FileInstantDataset”，默认为“QueueDataset”。


**参数**

    - **datafeed_class** (str) – datafeed类名，为QueueDataset或InMemoryDataset。默认为QueueDataset。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset()



