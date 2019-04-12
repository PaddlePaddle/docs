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


.. _cn_api_fluid_dataset_InMemoryDataset:

InMemoryDataset
-------------------------------

.. py:class:: paddle.fluid.dataset.InMemoryDataset

在InMemoryDataset中，它将数据加载到内存中并在训练之前重洗（shuffle）数据。


**代码示例**

.. code-block:: python

  dataset = paddle.fluid.DatasetFactory.create_dataset(“InMemoryDataset”)



.. py:method:: load_into_memory()

将数据加载到内存中。

**代码示例**

.. code-block:: python

  import paddle.fluid as fluid

  dataset = fluid.DatasetFactory.create_dataset("InMemoryDataset")
  filelist = ["a.txt", "b.txt"]
  dataset.set_filelist(filelist)
  dataset.load_into_memory()


.. py:method:: local_shuffle()

本地数据重洗。

**代码示例**

.. code-block:: python

  import paddle.fluid as fluid
  dataset = fluid.DatasetFactory.create_dataset("InMemoryDataset")
  filelist = ["a.txt", "b.txt"]
  dataset.set_filelist(filelist)
  dataset.local_shuffle()


.. py:method:: global_shuffle(fleet=None)

全局数据重洗。 Global shuffle仅适用于分布式模式， 即在单机或多机上同时进行训练的多个过程。 如果以分布式模式运行，则应传递fleet而不应传递None。


**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.incubate.fleet.parameter_server as fleet
    dataset = fluid.DatasetFactory.create_dataset("InMemoryDataset")
    filelist = ["a.txt", "b.txt"]
    dataset.set_filelist(filelist)
    dataset.global_shuffle(fleet)

参数: 
  - **fleet** – fleet单例，默认为None。




.. _cn_api_fluid_dataset_QueueDataset:


QueueDataset
-------------------------------

.. py:class:: paddle.fluid.dataset.QueueDataset

QueueDataset，可以流式处理数据。

**代码示例**

.. code-block:: python

  import paddle.fluid as fluid 
  dataset = fluid.DatasetFactory.create_dataset(“QueueDataset”)


.. py:method:: local_shuffle()

目前QueueDataset不支持本地数据重洗。



.. py:method:: global_shuffle(fleet=None)

全局数据重洗。

















