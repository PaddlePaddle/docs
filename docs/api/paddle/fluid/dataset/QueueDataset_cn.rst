.. _cn_api_fluid_dataset_QueueDataset:

QueueDataset
-------------------------------

.. py:class:: paddle.fluid.dataset.QueueDataset




流式处理数据。

代码示例
::::::::::::

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset("QueueDataset")



方法
::::::::::::
local_shuffle()
'''''''''

局域shuffle数据

QueueDataset中不支持局域shuffle，可能抛出NotImplementedError

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset("QueueDataset")
    dataset.local_shuffle()



global_shuffle(fleet=None)
'''''''''

全局shuffle数据

QueueDataset中不支持全局shuffle，可能抛出NotImplementedError

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet
    dataset = fluid.DatasetFactory().create_dataset("QueueDataset")
    dataset.global_shuffle(fleet)

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

set_batch_size(batch_size)
'''''''''

设置batch size。在训练期间生效。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_batch_size(128)

**参数**

    - **batch_size** (int) - batch size

set_fea_eval(record_candidate_size,fea_eval)
'''''''''

**参数**

    - **record_candidate_size** (int) - 打乱一个特征的候选实例大小
    - **fea_eval** (bool) - 是否设置特征验证模式来打乱特征，默认为True。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset(“InMemoryDataset”)
    dataset.set_fea_eval(1000000, True)

set_filelist(filelist)
'''''''''

在当前的worker中设置文件列表。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_filelist(["a.txt", "b.txt"])

**参数**

    - **filelist** (list) - 文件列表

set_hdfs_config(fs_name, fs_ugi)
'''''''''

设置hdfs配置：fs名称与ugi。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_hdfs_config("my_fs_name", "my_fs_ugi")

**参数**

    - **fs_name** (str) - fs名称
    - **fs_ugi** (str) - fs ugi

set_pipe_command(pipe_coommand)
'''''''''

在当前的 ``dataset`` 中设置pipe命令。pipe命令只能使用UNIX的pipe命令

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_pipe_command("python my_script.py")

**参数**

    - **pipe_command** (str) - pipe命令

set_thread(thread_num)
'''''''''

设置进程数量，等于readers的数量。

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

该方法是在特征层次上的一个打乱方法，经常被用在有着较大缩放率实例的稀疏矩阵上，为了比较metric，比如auc，在一个或者多个有着baseline的特征上做特征打乱来验证特征level的重要性。

**参数**

    - **slots** (list[string]) - 要打乱特征的集合

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset(“InMemoryDataset”)
    dataset.set_merge_by_lineid()
    #支持slot 0
    dataset.slots_shuffle([‘0’])

