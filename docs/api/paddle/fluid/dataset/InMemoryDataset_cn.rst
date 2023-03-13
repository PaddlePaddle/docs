.. _cn_api_fluid_dataset_InMemoryDataset:

InMemoryDataset
-------------------------------

.. py:class:: paddle.fluid.dataset.InMemoryDataset




InMemoryDataset 会向内存中加载数据并在训练前缓冲数据。此类由 DatasetFactory 创建。

代码示例
::::::::::::

.. code-block:: python

    dataset = paddle.fluid.DatasetFactory().create_dataset(“InMemoryDataset”)

方法
::::::::::::
set_queue_num(queue_num)
'''''''''

设置 ``Dataset`` 输出队列数量，训练进程会从队列中获取数据。

**参数**

    - **queue_num** (int) - dataset 输出队列数量

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
    dataset.set_queue_num(12)

set_fleet_send_batch_size(fleet_send_batch_size)
'''''''''

设置发送 batch 的大小

**参数**

    - **fleet_send_batch_size** (int) - 设置发送 batch 的大小。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
    dataset.set_fleet_send_batch_size(800)

set_merge_by_lineid(var_list, erase_duplicate_feas=True, min_merge_size=2, keep_unmerged-ins=True)
'''''''''

通过样本 id 来设置合并，一些线 id 的实例将会在 shuffle 之后进行合并，你应该在一个 data 生成器里面解析样本 id。

**参数**

    - **var_list** (list) - 可以被合并的特征列表，其中的每一个元素都是一个 ``Variable``。一些类特征我们通常不把它们合并为同样的样本 id，所以用户应当指定哪个类特征可以被合并。
    - **erase_duplicate_feas** (bool) - 合并的时候是否删除重复的特征值。默认为 True。
    - **min_merge_size** (int) - 合并的最小数量。默认为 2。
    - **keep_unmerged_ins** (bool) - 是否保留没有合并的样本，比如有着独特 id 的样本，或者重复 id 的数量小于 ``min_merge_size`` 的样本。

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
    dataset.set_merge_by_lineid()

load_into_memory()
'''''''''

向内存中加载数据。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
    filelist = ["a.txt", "b.txt"]
    dataset.set_filelist(filelist)
    dataset.load_into_memory()

preload_into_memory()
'''''''''

向内存中以异步模式加载数据。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
    filelist = ["a.txt", "b.txt"]
    dataset.set_filelist(filelist)
    dataset.preload_into_memory()
    dataset.wait_preload_done()

wait_preload_done()
'''''''''

等待 ``preload_into_memory`` 完成。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
    filelist = ["a.txt", "b.txt"]
    dataset.set_filelist(filelist)
    dataset.preload_into_memory()
    dataset.wait_preload_done()

local_shuffle()
'''''''''

局域 shuffle。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
    filelist = ["a.txt", "b.txt"]
    dataset.set_filelist(filelist)
    dataset.load_into_memory()
    dataset.local_shuffle()


global_shuffle(fleet=None)
'''''''''

全局 shuffle。

只能用在分布式模式（单机多进程或多机多进程）中。您如果在分布式模式中运行，应当传递 fleet 而非 None。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet
    dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
    filelist = ["a.txt", "b.txt"]
    dataset.set_filelist(filelist)
    dataset.load_into_memory()
    dataset.global_shuffle(fleet)

**参数**

    - **fleet** (Fleet) – fleet 单例。默认为 None。


release_memory()
'''''''''

当数据不再使用时，释放 InMemoryDataset 内存数据。

**代码示例**

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

get_memory_data_size(fleet=None)
'''''''''

用户可以调用此函数以了解加载进内存后所有 workers 中的样本数量。

.. note::
    该函数可能会导致性能不佳，因为它具有 barrier。

**参数**

    - **fleet** (Fleet) – fleet 对象。

**返回**
内存数据的大小。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet
    dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
    filelist = ["a.txt", "b.txt"]
    dataset.set_filelist(filelist)
    dataset.load_into_memory()
    print dataset.get_memory_data_size(fleet)


get_shuffle_data_size(fleet=None)
'''''''''

获取 shuffle 数据大小，用户可以调用此函数以了解局域/全局 shuffle 后所有 workers 中的样本数量。

.. note::
    该函数可能会导致局域 shuffle 性能不佳，因为它具有 barrier。但其不影响局域 shuffle。

**参数**

    - **fleet** (Fleet) – fleet 对象。

**返回**
shuffle 数据的大小。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet
    dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
    filelist = ["a.txt", "b.txt"]
    dataset.set_filelist(filelist)
    dataset.load_into_memory()
    dataset.global_shuffle(fleet)
    print dataset.get_shuffle_data_size(fleet)


set_batch_size(batch_size)
'''''''''

设置 batch size。在训练期间生效。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_batch_size(128)

**参数**

    - **batch_size** (int) - batch size

set_fea_eval(record_candidate_size, fea_eval=True)
'''''''''

设置特征打乱特征验证模式，来修正特征 level 的重要性，特征打乱需要 ``fea_eval`` 被设置为 True。

**参数**

    - **record_candidate_size** (int) - 打乱一个特征的候选实例大小
    - **fea_eval** (bool) - 是否设置特征验证模式来打乱特征，默认为 True。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset(“InMemoryDataset”)
    dataset.set_fea_eval(1000000, True)

desc()
'''''''''

为 ``DataFeedDesc`` 返回一个缓存信息。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset()
    print(dataset.desc())

**返回**
一个字符串信息

set_filelist(filelist)
'''''''''

在当前的 worker 中设置文件列表。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_filelist(["a.txt", "b.txt"])

**参数**

    - **filelist** (list) - 文件列表

set_hdfs_config(fs_name, fs_ugi)
'''''''''

设置 hdfs 配置：fs 名称与 ugi。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_hdfs_config("my_fs_name", "my_fs_ugi")

**参数**

    - **fs_name** (str) - fs 名称
    - **fs_ugi** (str) - fs ugi

set_pipe_command(pipe_coommand)
'''''''''

在当前的 ``dataset`` 中设置 pipe 命令。pipe 命令只能使用 UNIX 的 pipe 命令

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_pipe_command("python my_script.py")

**参数**

    - **pipe_command** (str) - pipe 命令

set_thread(thread_num)
'''''''''

设置进程数量，等于 readers 的数量。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_thread(12)

**参数**

    - **thread_num** (int) - 进程数量

set_use_var(var_list)
'''''''''

设置将要使用的 ``Variable`` 。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_use_var([data, label])

**参数**

    - **var_list** (list) - variable 列表

slots_shuffle(slots)
'''''''''

该方法是在特征层次上的一个打乱方法，经常被用在有着较大缩放率实例的稀疏矩阵上，为了比较 metric，比如 auc，在一个或者多个有着 baseline 的特征上做特征打乱来验证特征 level 的重要性。

**参数**

    - **slots** (list[string]) - 要打乱特征的集合

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset(“InMemoryDataset”)
    dataset.set_merge_by_lineid()
    #支持 slot 0
    dataset.slots_shuffle([‘0’])
