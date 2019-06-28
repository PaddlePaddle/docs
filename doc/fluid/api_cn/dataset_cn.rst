#################
 fluid.dataset
#################






.. _cn_api_fluid_dataset_DatasetFactory:

DatasetFactory
-------------------------------

.. py:class:: paddle.fluid.dataset.DatasetFactory

DatasetFactory是一个按数据集名称创建数据集的 "工厂"，可以创建“QueueDataset”，“InMemoryDataset”或“FileInstantDataset”，默认为“QueueDataset”。


**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    dataset = paddle.fluid.DatasetFactory().create_dataset("InMemoryDataset")

.. py:method:: create_dataset(datafeed_class='QueueDataset')

创建“QueueDataset”，“InMemoryDataset” 或 “FileInstantDataset”，默认为“QueueDataset”。


参数：
    - **datafeed_class** (str) – datafeed类名，为QueueDataset或InMemoryDataset。默认为QueueDataset。

**代码示例**:

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset()



.. _cn_api_fluid_dataset_InMemoryDataset:

InMemoryDataset
-------------------------------

.. py:class:: paddle.fluid.dataset.InMemoryDataset

InMemoryDataset会向内存中加载数据并在训练前缓冲数据。此类由DatasetFactory创建。

**代码示例**:

.. code-block:: python

    dataset = paddle.fluid.DatasetFactory().create_dataset(“InMemoryDataset”)


.. py:method:: load_into_memory()

向内存中加载数据。

**代码示例**:

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
    filelist = ["a.txt", "b.txt"]
    dataset.set_filelist(filelist)
    dataset.load_into_memory()


.. py:method:: local_shuffle()

局域shuffle。

**代码示例**:

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
    filelist = ["a.txt", "b.txt"]
    dataset.set_filelist(filelist)
    dataset.load_into_memory()
    dataset.local_shuffle()


.. py:method:: global_shuffle(fleet=None)

全局shuffle。

只能用在分布式模式（单机多进程或多机多进程）中。您如果在分布式模式中运行，应当传递fleet而非None。

**代码示例**:

.. code-block:: python

    import paddle.fluid as fluid
    from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet
    dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
    filelist = ["a.txt", "b.txt"]
    dataset.set_filelist(filelist)
    dataset.load_into_memory()
    dataset.global_shuffle(fleet)

参数：
    - **fleet** (Fleet) – fleet单例。默认为None。


.. py:method:: release_memory()

当数据不再使用时，释放InMemoryDataset内存数据。

**代码示例**:

.. code-block:: python

    import paddle.fluid as fluid
    from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet
    dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
    filelist = ["a.txt", "b.txt"]
    dataset.set_filelist(filelist)
    dataset.load_into_memory()
    dataset.global_shuffle(fleet)
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    exe.train_from_dataset(fluid.default_main_program(), dataset)dataset.release_memory()
    dataset.release_memory()

.. py:method:: get_memory_data_size(fleet=None)

用户可以调用此函数以了解加载进内存后所有workers中的ins数量。

.. note::
    该函数可能会导致性能不佳，因为它具有barrier。

参数：
    - **fleet** (Fleet) – fleet对象。

返回：内存数据的大小。

**代码示例**:

.. code-block:: python

    import paddle.fluid as fluid
    from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet
    dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
    filelist = ["a.txt", "b.txt"]
    dataset.set_filelist(filelist)
    dataset.load_into_memory()
    print dataset.get_memory_data_size(fleet)


.. py:method:: get_shuffle_data_size(fleet=None)

获取shuffle数据大小，用户可以调用此函数以了解局域/全局shuffle后所有workers中的ins数量。

.. note::
    该函数可能会导致局域shuffle性能不佳，因为它具有barrier。但其不影响局域shuffle。

参数：
    - **fleet** (Fleet) – fleet对象。

返回：shuffle数据的大小。

**代码示例**:

.. code-block:: python

    import paddle.fluid as fluid
    from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet
    dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
    filelist = ["a.txt", "b.txt"]
    dataset.set_filelist(filelist)
    dataset.load_into_memory()
    dataset.global_shuffle(fleet)
    print dataset.get_shuffle_data_size(fleet)




.. _cn_api_fluid_dataset_QueueDataset:

QueueDataset
-------------------------------

.. py:class:: paddle.fluid.dataset.QueueDataset

流式处理数据。

**代码示例**:

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset("QueueDataset")



.. py:method:: local_shuffle()

局域shuffle数据

QueueDataset中不支持局域shuffle，可能抛出NotImplementedError

**代码示例**:

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset("QueueDataset")
    dataset.local_shuffle()



.. py:method:: global_shuffle(fleet=None)

全局shuffle数据

QueueDataset中不支持全局shuffle，可能抛出NotImplementedError

**代码示例**:

.. code-block:: python

    import paddle.fluid as fluid
    from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet
    dataset = fluid.DatasetFactory().create_dataset("QueueDataset")
    dataset.global_shuffle(fleet)

