
..  _cluster_quick_start_ps:

快速开始-参数服务器
-------------------------

搜索推荐场景经常面临两个问题：

1. 海量训练数据：单机训练太慢，需要增加训练节点数。
2. 特征维度高且稀疏化：模型稀疏参数过多，单机内存无法容纳，需要采用分布式存储。

参数服务器（ParameterServer）模式采用了一种将模型参数中心化管理的方式来实现模型参数的分布式存储和更新。该模式下的节点/进程有两种不同的角色：

1. 训练节点（Trainer/Worker）：该节点负责完成数据读取、从服务节点拉取参数、前向计算、反向梯度计算等过程，并将计算出的梯度上传至服务节点。
2. 服务节点（Server）：在收到所有训练节点传来的梯度后，该节点会将梯度聚合并更新参数，供训练节点拉取进行下一轮的训练。

因此参数服务器模式对于存储超大规模模型参数的训练场景十分友好，常被用于训练拥有海量稀疏参数的搜索推荐领域模型。

1.1 任务介绍
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

本节将采用推荐领域非常经典的模型 wide_and_deep 为例，介绍如何使用飞桨分布式完成参数服务器训练任务。

参数服务器训练基于飞桨静态图，为方便用户理解，我们准备了一个 wide_and_deep 模型的单机静态图示例：\ `单机静态图示例 <https://github.com/PaddlePaddle/PaddleFleetX/tree/old_develop/eval/rec/wide_and_deep_single_static>`_\。

在单机静态图示例基础上，通过 1.2 章节的操作方法，可以将其修改为参数服务器训练示例，本次快速开始的完整示例代码参考：\ `参数服务器完整示例 <https://github.com/PaddlePaddle/PaddleFleetX/tree/old_develop/examples/wide_and_deep_dataset>`_\。

同时，我们在 AIStudio 上建立了一个参数服务器快速开始的项目：\ `参数服务器快速开始 <https://aistudio.baidu.com/aistudio/projectdetail/4522337>`_\，用户可以跳转到 AIStudio 上直接运行参数服务器的训练代码。

在编写分布式训练程序之前，用户需要确保已经安装 PaddlePaddle2.3 及以上版本的飞桨开源框架。

1.2 操作方法
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

参数服务器训练的基本代码主要包括如下几个部分：

    1. 导入分布式训练需要的依赖包。
    2. 定义分布式模式并初始化分布式训练环境。
    3. 加载模型。
    4. 构建 dataset 加载数据
    5. 定义参数更新策略及优化器。
    6. 开始训练。


下面将逐一进行讲解。

1.2.1 导入依赖
""""""""""""

导入必要的依赖，例如分布式训练专用的 Fleet API(paddle.distributed.fleet)。

.. code-block:: python

    import paddle
    import paddle.distributed.fleet as fleet

1.2.2 定义分布式模式并初始化分布式训练环境
""""""""""""

通过 ``fleet.init()`` 接口，用户可以定义训练相关的环境，注意此环境是用户预先在环境变量中配置好的，包括：训练节点个数，服务节点个数，当前节点的序号，服务节点完整的 IP:PORT 列表等。

.. code-block:: python

    # 当前参数服务器模式只支持静态图模式， 因此训练前必须指定 ``paddle.enable_static()``
    paddle.enable_static()
    fleet.init(is_collective=False)

1.2.3 加载模型
""""""""""""

.. code-block:: python

    # 模型定义参考 examples/wide_and_deep_dataset 中 model.py
    from model import WideDeepModel
    model = WideDeepModel()
    model.net(is_train=True)

1.2.4 构建 dataset 加载数据
""""""""""""

由于搜索推荐场景涉及到的训练数据通常较大，为提升训练中的数据读取效率，参数服务器采用 InMemoryDataset/QueueDataset 进行高性能的 IO。

InMemoryDataset/QueueDataset 所对应的数据处理脚本参考 examples/wide_and_deep_dataset/reader.py，与单机 DataLoader 相比，存在如下区别：

    1. 继承自 ``fleet.MultiSlotDataGenerator`` 基类。
    2. 复用单机 reader 中的 ``line_process()`` 方法，该方法将数据文件中一行的数据处理后生成特征数组并返回，特征数组不需要转成 np.array 格式。
    3. 实现基类中的 ``generate_sample()`` 函数，调用 ``line_process()`` 方法逐行读取数据进行处理，并返回一个可以迭代的 reader 方法。
    4. reader 方法需返回一个 list，其中的每个元素都是一个元组，具体形式为 ``(特征名，[特征值列表])`` ，元组的第一个元素为特征名（string 类型，需要与模型中对应输入 input 的 name 对应），第二个元素为特征值列表（list 类型）。
    5. 在__main__作用域中调用 ``run_from_stdin()`` 方法，直接从标准输入流获取待处理数据，而不需要对数据文件进行操作。

一个完整的 reader.py 伪代码如下：

.. code-block:: python

    import paddle
    # 导入所需要的 fleet 依赖
    import paddle.distributed.fleet as fleet

    # 需要继承 fleet.MultiSlotDataGenerator
    class WideDeepDatasetReader(fleet.MultiSlotDataGenerator):
        def line_process(self, line):
            features = line.rstrip('\n').split('\t')
            # 省略数据处理过程，具体实现可参考单机 reader 的 line_process()方法
            # 返回值为一个 list，其中的每个元素均为一个 list，不需要转成 np.array 格式
            # 具体格式：[[dense_value1, dense_value2, ...], [sparse_value1], [sparse_value2], ..., [label]]
            return [dense_feature] + sparse_feature + [label]

        # 实现 generate_sample()函数
        # 该方法有一个名为 line 的参数，只需要逐行处理数据，不需要对数据文件进行操作
        def generate_sample(self, line):
            def wd_reader():
                # 按行处理数据
                input_data = self.line_process(line)

                # 构造特征名数组 feature_name
                feature_name = ["dense_input"]
                for idx in categorical_range_:
                    feature_name.append("C" + str(idx - 13))
                feature_name.append("label")

                # 返回一个 list，其中的每个元素都是一个元组
                # 元组的第一个元素为特征名（string 类型），第二个元素为特征值（list 类型）
                # 具体格式：[('dense_input', [dense_value1, dense_value2, ...]), ('C1', [sparse_value1]), ('C2', [sparse_value2]), ..., ('label', [label])]
                yield zip(feature_name, input_data)

            # generate_sample()函数需要返回一个可以迭代的 reader 方法
            return wd_reader

    if __name__ == "__main__":
        # 调用 run_from_stdin()方法，直接从标准输入流获取待处理数据
        my_data_generator = WideDeepDatasetReader()
        my_data_generator.run_from_stdin()

在训练脚本中，构建 dataset 加载数据：

.. code-block:: python

    dataset = paddle.distributed.QueueDataset()
    thread_num = 1

    # use_var 指定网络中的输入数据，pipe_command 指定数据处理脚本
    # 要求 use_var 中输入数据的顺序与数据处理脚本输出的特征顺序一一对应
    dataset.init(use_var=model.inputs,
                 pipe_command="python reader.py",
                 batch_size=batch_size,
                 thread_num=thread_num)

    train_files_list = [os.path.join(train_data_path, x)
                          for x in os.listdir(train_data_path)]

    # set_filelist 指定 dataset 读取的训练文件的列表
    dataset.set_filelist(train_files_list)

备注：dataset 更详细用法参见\ `使用 InMemoryDataset/QueueDataset 进行训练 <https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/parameter_server/performance/dataset.html>`_\。


1.2.5 定义参数更新策略及优化器
""""""""""""

在 Fleet API 中，用户可以使用 ``fleet.DistributedStrategy()`` 接口定义自己想要使用的分布式策略。

其中 ``a_sync`` 选项用于定义参数服务器相关的策略，当其被设定为 ``False`` 时，分布式训练将在同步的模式下进行。反之，当其被设定成 ``True`` 时，分布式训练将在异步的模式下进行。

.. code-block:: python

    # 定义异步训练
    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.a_sync = True

用户需要使用 ``fleet.distributed_optimizer()`` 接口，将单机优化器转换成分布式优化器，并最小化模型的损失值。

.. code-block:: python

    # 定义单机优化器
    optimizer = paddle.optimizer.SGD(learning_rate=0.0001)
    # 单机优化器转换成分布式优化器
    optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
    # 使用分布式优化器最小化模型损失值 model.loss，model.loss 定义参见 model.py
    optimizer.minimize(model.loss)

1.2.6 开始训练
""""""""""""

完成模型及训练策略以后，我们就可以开始训练模型了。因为在参数服务器模式下会有不同的角色，所以根据不同节点分配不同的任务。

对于服务器节点，首先用 ``init_server()`` 接口对其进行初始化，然后启动服务并开始监听由训练节点传来的梯度。

同样对于训练节点，用 ``init_worker()`` 接口进行初始化后， 开始执行训练任务。运行 ``exe.train_from_dataset()`` 接口开始训练。

.. code-block:: python

    if fleet.is_server():
        fleet.init_server()
        fleet.run_server()
    else:
        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(paddle.static.default_startup_program())

        fleet.init_worker()

        for epoch_id in range(1):
            exe.train_from_dataset(paddle.static.default_main_program(),
                                   dataset,
                                   paddle.static.global_scope(),
                                   debug=False,
                                   fetch_list=[model.loss],
                                   fetch_info=["loss"],
                                   print_period=1)

        fleet.stop_worker()

备注：Paddle2.3 版本及以后，ParameterServer 训练将废弃掉 dataloader + exe.run()方式，请切换到 dataset + exe.train_from_dataset()方式。


1.3 运行训练脚本
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

定义完训练脚本后，我们就可以用 ``fleetrun`` 指令运行分布式任务了。 ``fleetrun`` 是飞桨封装的分布式启动命令，命令参数 ``server_num`` , ``worker_num`` 分别为服务节点和训练节点的数量。在本例中，服务节点有 1 个，训练节点有 2 个。

.. code-block:: bash

    fleetrun --server_num=1 --trainer_num=2 train.py

您将在执行终端看到如下日志信息：

.. code-block:: bash

    LAUNCH INFO 2022-05-18 11:27:17,761 -----------  Configuration  ----------------------
    LAUNCH INFO 2022-05-18 11:27:17,761 devices: None
    LAUNCH INFO 2022-05-18 11:27:17,761 elastic_level: -1
    LAUNCH INFO 2022-05-18 11:27:17,761 elastic_timeout: 30
    LAUNCH INFO 2022-05-18 11:27:17,761 gloo_port: 6767
    LAUNCH INFO 2022-05-1811:27:17,761 host: None
    LAUNCH INFO 2022-05-18 11:27:17,761 job_id: default
    LAUNCH INFO 2022-05-18 11:27:17,761 legacy: False
    LAUNCH INFO 2022-05-18 11:27:17,761 log_dir: log
    LAUNCH INFO 2022-05-18 11:27:17,761 log_level: INFO
    LAUNCH INFO 2022-05-18 11:27:17,762 master: None
    LAUNCH INFO 2022-05-18 11:27:17,762 max_restart: 3
    LAUNCH INFO 2022-05-18 11:27:17,762 nnodes: 1
    LAUNCH INFO 2022-05-18 11:27:17,762 nproc_per_node: None
    LAUNCH INFO 2022-05-18 11:27:17,762 rank: -1
    LAUNCH INFO 2022-05-18 11:27:17,762 run_mode: collective
    LAUNCH INFO 2022-05-18 11:27:17,762 server_num: 1
    LAUNCH INFO 2022-05-18 11:27:17,762 servers:
    LAUNCH INFO 2022-05-18 11:27:17,762 trainer_num: 2
    LAUNCH INFO 2022-05-18 11:27:17,762 trainers:
    LAUNCH INFO 2022-05-18 11:27:17,762 training_script: train.py
    LAUNCH INFO 2022-05-18 11:27:17,762 training_script_args: []
    LAUNCH INFO 2022-05-18 11:27:17,762 with_gloo: 0
    LAUNCH INFO 2022-05-18 11:27:17,762 --------------------------------------------------
    LAUNCH INFO 2022-05-18 11:27:17,772 Job: default, mode ps, replicas 1[1:1], elastic False
    LAUNCH INFO 2022-05-18 11:27:17,775 Run Pod: evjsyn, replicas 3, status ready
    LAUNCH INFO 2022-05-18 11:27:17,795 Watching Pod: evjsyn, replicas 3, status running

同时，在 log 目录下，会生成服务节点和训练节点的日志文件。
服务节点日志：default.evjsyn.ps.0.log，日志中须包含以下内容，证明服务节点启动成功，可以提供服务。

.. code-block:: bash

    I0518 11:27:20.730531 177420 brpc_ps_server.cc:73] running server with rank id: 0, endpoint: IP:PORT

训练节点日志：default.evjsyn.trainer.0.log，日志中打印了训练过程中的部分变量值。

.. code-block:: bash

    time: [2022-05-18 11:27:27], batch: [1], loss[1]:[0.666739]
    time: [2022-05-18 11:27:27], batch: [2], loss[1]:[0.690405]
    time: [2022-05-18 11:27:27], batch: [3], loss[1]:[0.681693]
    time: [2022-05-18 11:27:27], batch: [4], loss[1]:[0.703863]
    time: [2022-05-18 11:27:27], batch: [5], loss[1]:[0.670717]

备注：启动相关问题，请参考 :ref:`cn_api_distributed_launch` 。
