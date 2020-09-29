.. _cn_api_distributed_InMemoryDataset:

InMemoryDataset
-------------------------------


.. py:class:: paddle.distributed.InMemoryDataset




InMemoryDataset会根据用户自定义的预处理指令预处理原始数据，向内存中加载数据并在训练前缓冲数据。此类由paddle.distributed.InMemoryDataset直接创建。

**代码示例**:

.. code-block:: python

    import paddle
    paddle.enable_static()
    dataset = paddle.distributed.InMemoryDataset()

.. py:method:: init(**kwargs)

**注意：**

  **1. 该API只在非** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

对InMemoryDataset的实例进行配置初始化。

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

    dataset = paddle.distributed.InMemoryDataset()
    dataset.init(
        batch_size=1,
        thread_num=2,
        input_type=1,
        pipe_command="cat",
        use_var=slots_vars)
    dataset.set_filelist(
        ["test_queue_dataset_run_a.txt", "test_queue_dataset_run_b.txt"])
    dataset.load_into_memory()

    paddle.enable_static()
    
    place = paddle.CPUPlace()
    exe = paddle.static.Executor(place)
    startup_program = paddle.static.Program()
    main_program = paddle.static.Program()
    exe.run(startup_program)

    exe.train_from_dataset(main_program, dataset)
    
    os.remove("./test_queue_dataset_run_a.txt")
    os.remove("./test_queue_dataset_run_b.txt")

.. py:method:: _init_distributed_settings(**kwargs)

**注意：**

  **1. 该API只在非** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**
  **2. 本api需要在机大规模参数服务器训练下生效，敬请期待详细使用文档**

对InMemoryDataset的实例进行分布式训俩相关配置的初始化。

参数：
    - **kwargs** - 可选的关键字参数，由调用者提供， 目前支持以下关键字配置。
    - **merge_size** (int) - 通过样本id来设置合并，相同id的样本将会在shuffle之后进行合并，你应该在一个data生成器里面解析样本id。merge_size表示合并的最小数量，默认值为-1，表示不做合并。
    - **parse_ins_id** (bool) - 是否需要解析每条样的id，默认值为False。
    - **parse_content** (bool) - 是否需要解析每条样本的content, 默认值为False。
    - **fleet_send_batch_size** (int) - 设置发送batch的大小，默认值为1024。
    - **fleet_send_sleep_seconds** (int) - 设置发送batch后的睡眠时间，默认值为0。
    - **fea_eval** (bool) - 设置特征打乱特征验证模式，来修正特征级别的重要性， 特征打乱需要 ``fea_eval`` 被设置为True. 默认值为False。
    - **candidate_size** (int) - 特征打乱特征验证模式下，用于随机化特征的候选池大小. 默认值为10000。

返回：None。


**代码示例**

.. code-block:: python

    import paddle

    paddle.enable_static()

    dataset = paddle.distributed.InMemoryDataset()
    dataset.init(
        batch_size=1,
        thread_num=2,
        input_type=1,
        pipe_command="cat",
        use_var=[])
    dataset._init_distributed_settings(
        parse_ins_id=True,
        parse_content=True,
        fea_eval=True,
        candidate_size=10000)


.. py:method:: update_settings(**kwargs)

**注意：**

  **1. 该API只在非** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

对InMemoryDataset的实例通过init和_init_distributed_settings初始化的配置进行更新。

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
    - **merge_size** (int) - 通过样本id来设置合并，相同id的样本将会在shuffle之后进行合并，你应该在一个data生成器里面解析样本id。merge_size表示合并的最小数量，默认值为-1，表示不做合并。
    - **parse_ins_id** (bool) - 是否需要解析每条样的id，默认值为False。
    - **parse_content** (bool) 是否需要解析每条样本的content, 默认值为False。
    - **fleet_send_batch_size** (int) - 设置发送batch的大小，默认值为1024。
    - **fleet_send_sleep_seconds** (int) - 设置发送batch后的睡眠时间，默认值为0。
    - **fea_eval** (bool) - 设置特征打乱特征验证模式，来修正特征级别的重要性， 特征打乱需要 ``fea_eval`` 被设置为True. 默认值为False。
    - **candidate_size** (int) - 特征打乱特征验证模式下，用于随机化特征的候选池大小. 默认值为10000。

返回：None。


**代码示例**

.. code-block:: python

    import paddle
    
    paddle.enable_static()

    dataset = paddle.distributed.InMemoryDataset()
    dataset.init(
        batch_size=1,
        thread_num=2,
        input_type=1,
        pipe_command="cat",
        use_var=[])
    dataset._init_distributed_settings(
        parse_ins_id=True,
        parse_content=True,
        fea_eval=True,
        candidate_size=10000)
    dataset.update_settings(batch_size=2)


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
    dataset = paddle.distributed.InMemoryDataset()
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

.. py:method:: load_into_memory()

**注意：**

  **1. 该API只在非** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

向内存中加载数据。

**代码示例**:

.. code-block:: python

    import paddle

    paddle.enable_static()
    
    dataset = paddle.distributed.InMemoryDataset()
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
    dataset.load_into_memory()

.. py:method:: preload_into_memory(thread_num=None)

向内存中以异步模式加载数据。

参数：
    - **thread_num** (int) - 异步加载数据时的线程数。

**代码示例**:

.. code-block:: python

    import paddle

    paddle.enable_static()

    dataset = paddle.distributed.InMemoryDataset()
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
    dataset.preload_into_memory()
    dataset.wait_preload_done()

.. py:method:: wait_preload_done()

等待 ``preload_into_memory`` 完成。

**代码示例**:

.. code-block:: python

    import paddle

    paddle.enable_static()

    dataset = paddle.distributed.InMemoryDataset()
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
    dataset.preload_into_memory()
    dataset.wait_preload_done()

.. py:method:: local_shuffle()

局域shuffle。加载到内存的训练样本进行单机节点内部的打乱

**代码示例**:

.. code-block:: python

    import paddle

    paddle.enable_static()

    dataset = paddle.distributed.InMemoryDataset()
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
    dataset.load_into_memory()
    dataset.local_shuffle()

.. py:method:: global_shuffle(fleet=None, thread_num=12)

全局shuffle。

只能用在分布式模式（单机多进程或多机多进程）中。您如果在分布式模式中运行，应当传递fleet而非None。

**代码示例**:

.. code-block:: python

    import paddle

    paddle.enable_static()

    dataset = paddle.distributed.InMemoryDataset()
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
    dataset.load_into_memory()
    dataset.global_shuffle()

参数：
    - **fleet** (Fleet) – fleet单例。默认为None。
    - **thread_num** (int) - 全局shuffle时的线程数。

.. py:method:: release_memory()

当数据不再使用时，释放InMemoryDataset内存数据。

**代码示例**:

.. code-block:: python

    import paddle

    paddle.enable_static()
    
    dataset = paddle.distributed.InMemoryDataset()
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
    dataset.load_into_memory()
    dataset.global_shuffle()
    exe = paddle.static.Executor(paddle.CPUPlace())
    startup_program = paddle.static.Program()
    main_program = paddle.static.Program()
    exe.run(startup_program)
    exe.train_from_dataset(main_program, dataset)
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

    import paddle

    paddle.enable_static()

    dataset = paddle.distributed.InMemoryDataset()
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
    dataset.load_into_memory()
    print dataset.get_memory_data_size()


.. py:method:: get_shuffle_data_size(fleet=None)

获取shuffle数据大小，用户可以调用此函数以了解局域/全局shuffle后所有workers中的样本数量。

.. note::
    该函数可能会导致局域shuffle性能不佳，因为它具有barrier。但其不影响局域shuffle。

参数：
    - **fleet** (Fleet) – fleet对象。

返回：shuffle数据的大小。

**代码示例**:

.. code-block:: python

    import paddle

    paddle.enable_static()
    
    dataset = paddle.distributed.InMemoryDataset()
    dataset = paddle.distributed.InMemoryDataset()
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
    dataset.load_into_memory()
    dataset.global_shuffle()
    print dataset.get_shuffle_data_size()

.. py:method:: slots_shuffle(slots)

该方法是在特征层次上的一个打乱方法，经常被用在有着较大缩放率实例的稀疏矩阵上，为了比较metric，比如auc，在一个或者多个有着baseline的特征上做特征打乱来验证特征level的重要性。

参数：
    - **slots** (list[string]) - 要打乱特征的集合

**代码示例**:

.. code-block:: python

    import paddle

    paddle.enable_static()
    
    dataset = paddle.distributed.InMemoryDataset()
    dataset._init_distributed_settings(fea_eval=True)
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
    dataset.load_into_memory()
    dataset.slots_shuffle(['slot1'])



