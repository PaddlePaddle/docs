.. _cn_api_paddle_distributed_InMemoryDataset:

InMemoryDataset
-------------------------------


.. py:class:: paddle.distributed.InMemoryDataset()



InMemoryDataset，它将数据加载到内存中，并在训练前随机整理数据。

代码示例
::::::::::::

COPY-FROM: paddle.distributed.InMemoryDataset

方法
::::::::::::
init(**kwargs)
'''''''''

**注意：**

  **1. 该 API 只在非** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

对 InMemoryDataset 的实例进行配置初始化。

**参数**

    - **kwargs** - 可选的关键字参数，由调用者提供，目前支持以下关键字配置。
    - **batch_size** (int) - batch size 的大小。默认值为 1。
    - **thread_num** (int) - 用于训练的线程数，默认值为 1。
    - **use_var** (list) - 用于输入的 variable 列表，默认值为[]。
    - **input_type** (int) - 输入到模型训练样本的类型。0 代表一条样本，1 代表一个 batch。默认值为 0。
    - **fs_name** (str) - hdfs 名称。默认值为""。
    - **fs_ugi** (str) - hdfs 的 ugi。默认值为""。
    - **pipe_command** (str) - 在当前的 ``dataset`` 中设置的 pipe 命令用于数据的预处理。pipe 命令只能使用 UNIX 的 pipe 命令，默认为"cat"。
    - **download_cmd** (str) - 数据下载 pipe 命令。pipe 命令只能使用 UNIX 的 pipe 命令，默认为"cat"。


**返回**
None。


**代码示例**

COPY-FROM: paddle.distributed.InMemoryDataset.init

_init_distributed_settings(**kwargs)
'''''''''

**注意：**

  **1. 该 API 只在非** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**
  **2. 本 api 需要在机大规模参数服务器训练下生效，敬请期待详细使用文档**

对 InMemoryDataset 的实例进行分布式训练相关配置的初始化。

**参数**

    - **kwargs** - 可选的关键字参数，由调用者提供，目前支持以下关键字配置。
    - **merge_size** (int) - 通过样本 id 来设置合并，相同 id 的样本将会在 shuffle 之后进行合并，你应该在一个 data 生成器里面解析样本 id。merge_size 表示合并的最小数量，默认值为-1，表示不做合并。
    - **parse_ins_id** (bool) - 是否需要解析每条样的 id，默认值为 False。
    - **parse_content** (bool) - 是否需要解析每条样本的 content，默认值为 False。
    - **fleet_send_batch_size** (int) - 设置发送 batch 的大小，默认值为 1024。
    - **fleet_send_sleep_seconds** (int) - 设置发送 batch 后的睡眠时间，默认值为 0。
    - **fea_eval** (bool) - 设置特征打乱特征验证模式，来修正特征级别的重要性，特征打乱需要 ``fea_eval`` 被设置为 True。默认值为 False。
    - **candidate_size** (int) - 特征打乱特征验证模式下，用于随机化特征的候选池大小。默认值为 10000。

**返回**
None。


**代码示例**

COPY-FROM: paddle.distributed.InMemoryDataset._init_distributed_settings

update_settings(**kwargs)
'''''''''

**注意：**

  **1. 该 API 只在非** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

对 InMemoryDataset 的实例通过 init 和_init_distributed_settings 初始化的配置进行更新。

**参数**

    - **kwargs** - 可选的关键字参数，由调用者提供，目前支持以下关键字配置。
    - **batch_size** (int) - batch size 的大小。默认值为 1。
    - **thread_num** (int) - 用于训练的线程数，默认值为 1。
    - **use_var** (list) - 用于输入的 variable 列表，默认值为[]。
    - **input_type** (int) - 输入到模型训练样本的类型。0 代表一条样本，1 代表一个 batch。默认值为 0。
    - **fs_name** (str) - hdfs 名称。默认值为""。
    - **fs_ugi** (str) - hdfs 的 ugi。默认值为""。
    - **pipe_command** (str) - 在当前的 ``dataset`` 中设置的 pipe 命令用于数据的预处理。pipe 命令只能使用 UNIX 的 pipe 命令，默认为"cat"。
    - **download_cmd** (str) - 数据下载 pipe 命令。pipe 命令只能使用 UNIX 的 pipe 命令，默认为"cat"。
    - **merge_size** (int) - 通过样本 id 来设置合并，相同 id 的样本将会在 shuffle 之后进行合并，你应该在一个 data 生成器里面解析样本 id。merge_size 表示合并的最小数量，默认值为-1，表示不做合并。
    - **parse_ins_id** (bool) - 是否需要解析每条样的 id，默认值为 False。
    - **parse_content** (bool) 是否需要解析每条样本的 content，默认值为 False。
    - **fleet_send_batch_size** (int) - 设置发送 batch 的大小，默认值为 1024。
    - **fleet_send_sleep_seconds** (int) - 设置发送 batch 后的睡眠时间，默认值为 0。
    - **fea_eval** (bool) - 设置特征打乱特征验证模式，来修正特征级别的重要性，特征打乱需要 ``fea_eval`` 被设置为 True。默认值为 False。
    - **candidate_size** (int) - 特征打乱特征验证模式下，用于随机化特征的候选池大小。默认值为 10000。

**返回**
None。


**代码示例**

COPY-FROM: paddle.distributed.InMemoryDataset.update_settings

load_into_memory()
'''''''''

**注意：**

  **1. 该 API 只在非** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

向内存中加载数据。

**代码示例**

COPY-FROM: paddle.distributed.InMemoryDataset.load_into_memory

preload_into_memory(thread_num=None)
'''''''''

向内存中以异步模式加载数据。

**参数**

    - **thread_num** (int) - 异步加载数据时的线程数。

**代码示例**

COPY-FROM: paddle.distributed.InMemoryDataset.preload_into_memory

wait_preload_done()
'''''''''

等待 ``preload_into_memory`` 完成。

**代码示例**

COPY-FROM: paddle.distributed.InMemoryDataset.wait_preload_done

local_shuffle()
'''''''''

局部 shuffle。加载到内存的训练样本进行单机节点内部的打乱

**代码示例**

COPY-FROM: paddle.distributed.InMemoryDataset.local_shuffle

global_shuffle(fleet=None, thread_num=12)
'''''''''

全局 shuffle。只能用在分布式模式（单机多进程或多机多进程）中。您如果在分布式模式中运行，应当传递 fleet 而非 None。

**代码示例**

COPY-FROM: paddle.distributed.InMemoryDataset.global_shuffle

**参数**

    - **fleet** (Fleet) – fleet 单例。默认为 None。
    - **thread_num** (int) - 全局 shuffle 时的线程数。

release_memory()
'''''''''

当数据不再使用时，释放 InMemoryDataset 内存数据。

COPY-FROM: paddle.distributed.InMemoryDataset.release_memory

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

COPY-FROM: paddle.distributed.InMemoryDataset.get_memory_data_size

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

COPY-FROM: paddle.distributed.InMemoryDataset.get_shuffle_data_size

slots_shuffle(slots)
'''''''''

该方法是在特征层次上的一个打乱方法，经常被用在有着较大缩放率实例的稀疏矩阵上，为了比较 metric，比如 auc，在一个或者多个有着 baseline 的特征上做特征打乱来验证特征 level 的重要性。

**参数**

    - **slots** (list[string]) - 要打乱特征的集合

**代码示例**

COPY-FROM: paddle.distributed.InMemoryDataset.slots_shuffle
