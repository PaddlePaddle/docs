.. _cn_api_distributed_QueueDataset:

QueueyDataset
-------------------------------


.. py:class:: paddle.distributed.QueueDataset




QueueyDataset 是流式处理数据使用 Dataset 类。与 InmemoryDataset 继承自同一父类，用于单机训练，不支持分布式大规模参数服务器相关配置和 shuffle。此类由 paddle.distributed.QueueDataset 直接创建。

代码示例
::::::::::::

COPY-FROM: paddle.distributed.QueueDataset

方法
::::::::::::
init(**kwargs)
'''''''''

**注意：**

  **1. 该 API 只在非** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

对 QueueDataset 的实例进行配置初始化。

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

COPY-FROM: paddle.distributed.QueueDataset.init


set_filelist(filelist)
'''''''''

在当前的 worker 中设置文件列表。

**代码示例**

COPY-FROM: paddle.distributed.QueueDataset.set_filelist


**参数**

    - **filelist** (list[string]) - 文件列表
