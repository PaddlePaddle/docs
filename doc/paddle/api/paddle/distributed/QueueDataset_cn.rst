.. _cn_api_distributed_QueueDataset:

QueueyDataset
-------------------------------


.. py:class:: paddle.distributed.QueueDataset




QueueyDataset是流式处理数据使用Dataset类。与InmemoryDataset继承自同一父类，用于单机训练，不支持分布式大规模参数服务器相关配置和shuffle。此类由paddle.distributed.QueueDataset直接创建。

**代码示例**:

.. code-block:: python

    import paddle
    dataset = paddle.distributed.QueueDataset()

.. py:method:: init(**kwargs)

**注意：**

  **1. 该API只在非** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

对QueueDataset的实例进行配置初始化。

参数：
    - **kwargs** - 可选的关键字参数，由调用者提供， 目前支持以下关键字配置。
    - **batch_size** (int) - batch size的大小. 默认值为1。
    - **thread_num** (int) - 用于训练的线程数, 默认值为1。
    - **use_var** (list) - 用于输入的variable列表，默认值为[]。
    - **input_type** (int) - 输入到模型训练样本的类型. 0 代表一条样本, 1 代表一个batch。 默认值为0。
    - **fs_name** (str) - hdfs名称. 默认值为""。
    - **fs_ugi** (str) - hdfs的ugi. 默认值为""。
    - **pipe_command** (str) - 在当前的 ``dataset`` 中设置的pipe命令用于数据的预处理。pipe命令只能使用UNIX的pipe命令，默认为"cat"。
    - **download_cmd** (str) - 数据下载pipe命令。 pipe命令只能使用UNIX的pipe命令, 默认为"cat"。


返回：None。


**代码示例**

.. code-block:: python


    import paddle
    import os

    paddle.enable_static()

    with open("test_queue_dataset_run_a.txt", "w") as f:
        data = "2 1 2 2 5 4 2 2 7 2 1 3\n"
        data += "2 6 2 2 1 4 2 2 4 2 2 3\n"
        data += "2 5 2 2 9 9 2 2 7 2 1 3\n"
        data += "2 7 2 2 1 9 2 3 7 2 5 3\n"
        f.write(data)
    with open("test_queue_dataset_run_b.txt", "w") as f:
        data = "2 1 2 2 5 4 2 2 7 2 1 3\n"
        data += "2 6 2 2 1 4 2 2 4 2 2 3\n"
        data += "2 5 2 2 9 9 2 2 7 2 1 3\n"
        data += "2 7 2 2 1 9 2 3 7 2 5 3\n"
        f.write(data)

    slots = ["slot1", "slot2", "slot3", "slot4"]
    slots_vars = []
    for slot in slots:
        var = paddle.static.data(
            name=slot, shape=[None, 1], dtype="int64", lod_level=1)
        slots_vars.append(var)

    dataset = paddle.distributed.QueueDataset()
    dataset.init(
        batch_size=1,
        thread_num=2,
        input_type=1,
        pipe_command="cat",
        use_var=slots_vars)
    dataset.set_filelist(
        ["test_queue_dataset_run_a.txt", "test_queue_dataset_run_b.txt"])

    paddle.enable_static()
    
    place = paddle.CPUPlace()
    exe = paddle.static.Executor(place)
    startup_program = paddle.static.Program()
    main_program = paddle.static.Program()
    exe.run(startup_program)

    exe.train_from_dataset(main_program, dataset)
    
    os.remove("./test_queue_dataset_run_a.txt")
    os.remove("./test_queue_dataset_run_b.txt")


.. py:method:: set_filelist(filelist)

在当前的worker中设置文件列表。

**代码示例**:

.. code-block:: python

    import paddle
    import os

    paddle.enable_static()
    
    with open("test_queue_dataset_run_a.txt", "w") as f:
        data = "2 1 2 2 5 4 2 2 7 2 1 3\n"
        data += "2 6 2 2 1 4 2 2 4 2 2 3\n"
        data += "2 5 2 2 9 9 2 2 7 2 1 3\n"
        data += "2 7 2 2 1 9 2 3 7 2 5 3\n"
        f.write(data)
    with open("test_queue_dataset_run_b.txt", "w") as f:
        data = "2 1 2 2 5 4 2 2 7 2 1 3\n"
        data += "2 6 2 2 1 4 2 2 4 2 2 3\n"
        data += "2 5 2 2 9 9 2 2 7 2 1 3\n"
        data += "2 7 2 2 1 9 2 3 7 2 5 3\n"
        f.write(data)
    dataset = paddle.distributed.QueueDataset()
    slots = ["slot1", "slot2", "slot3", "slot4"]
    slots_vars = []
    for slot in slots:
        var = paddle.static.data(
            name=slot, shape=[None, 1], dtype="int64", lod_level=1)
        slots_vars.append(var)
    dataset.init(
        batch_size=1,
        thread_num=2,
        input_type=1,
        pipe_command="cat",
        use_var=slots_vars)
    filelist = ["a.txt", "b.txt"]
    dataset.set_filelist(filelist)
    os.remove("./test_queue_dataset_run_a.txt")
    os.remove("./test_queue_dataset_run_b.txt")


参数：
    - **filelist** (list[string]) - 文件列表
