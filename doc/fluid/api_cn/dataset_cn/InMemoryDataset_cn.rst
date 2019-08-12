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


.. py:method:: set_queue_num(queue_num)

设置 ``Dataset`` 的输出队列数量，训练进程从队列中获取数据。



参数:
    - **queue_num** (int) - dataset输出队列的数量。

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset") 
    dataset.set_queue_num(12)

.. py:method:: set_fleet_send_batch_size(fleet_send_batch_size)

设置发送batch的大小

参数:
    - **fleet_send_batch_size** (int) - 设置发送batch的大小。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
    dataset.set_fleet_send_batch_size(800)


.. py:method:: set_merge_by_lineid(var_list, erase_duplicate_feas=True, min_merge_size=2, keep_unmerged-ins=True)

通过样本id来设置合并，一些线id的实例将会在shuffle之后进行合并，你应该在一个data生成器里面解析样本id。

参数:
    - **var_list** (list) - 可以被合并的特征列表，其中的每一个元素都是一个 ``Variable`` 。一些类特征我们通常不把它们合并为同样的样本id，所以用户应当指定哪个类特征可以被合并。
    - **erase_duplicate_feas** (bool) - 合并的时候是否删除重复的特征值。默认为True。
    - **min_merge_size** (int) - 合并的最小数量。默认为2。
    - **keep_unmerged_ins** (bool) - 是否保留没有合并的样本，比如有着独特id的样本，或者重复id的数量小于 ``min_merge_size`` 的样本。

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
    exe.train_from_dataset(fluid.default_main_program(), dataset)
    dataset.release_memory()

.. py:method:: get_memory_data_size(fleet=None)

用户可以调用此函数以了解加载进内存后所有workers中的样本数量。

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

获取shuffle数据大小，用户可以调用此函数以了解局域/全局shuffle后所有workers中的样本数量。

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




