本节将采用推荐领域非常经典的模型wide_and_deep为例，介绍如何使用Fleet API（paddle.distributed.fleet）完成参数服务器训练任务，本次快速开始的完整示例代码位于 https://github.com/PaddlePaddle/FleetX/tree/develop/examples/wide_and_deep。

2.1 版本要求
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在编写分布式训练程序之前，用户需要确保已经安装paddlepaddle-2.0.0-rc-cpu或paddlepaddle-2.0.0-rc-gpu及以上版本的飞桨开源框架。

2.2 操作方法
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

参数服务器训练的基本代码主要包括如下几个部分：

    1. 导入分布式训练需要的依赖包。
    2. 定义分布式模式并初始化分布式训练环境。
    3. 加载模型及数据。
    4. 定义参数更新策略及优化器。
    5. 开始训练。 
    
下面将逐一进行讲解。

2.2.1 导入依赖
""""""""""""

导入必要的依赖，例如分布式训练专用的Fleet API(paddle.distributed.fleet)。

.. code-block:: python

    import paddle
    import paddle.distributed.fleet as fleet

2.2.2 定义分布式模式并初始化分布式训练环境
""""""""""""

通过 ``fleet.init()`` 接口，用户可以定义训练相关的环境，注意此环境是用户预先在环境变量中配置好的，包括：训练节点个数，服务节点个数，当前节点的序号，服务节点完整的IP:PORT列表等。

.. code-block:: python

    # 当前参数服务器模式只支持静态图模式， 因此训练前必须指定 ``paddle.enable_static()``
    paddle.enable_static()
    fleet.init(is_collective=False)

2.2.3 加载模型及数据
""""""""""""

.. code-block:: python

    # 模型定义参考 examples/wide_and_deep 中 model.py
    from model import WideDeepModel
    from reader import WideDeepDataset

    model = WideDeepModel()
    model.net(is_train=True)

    def distributed_training(exe, train_model, train_data_path="./data", batch_size=10, epoch_num=1):
        train_data = WideDeepDataset(data_path=train_data_path)
        reader = train_model.loader.set_sample_generator(
            train_data, batch_size=batch_size, drop_last=True, places=paddle.CPUPlace())

        for epoch_id in range(epoch_num):
            reader.start()
            try:
                while True:
                    loss_val = exe.run(program=paddle.static.default_main_program(),
                                    fetch_list=[train_model.cost.name])
                    loss_val = np.mean(loss_val)
                    print("TRAIN ---> pass: {} loss: {}\n".format(epoch_id, loss_val))
            except paddle.common_ops_import.core.EOFException:
                reader.reset()

    
    
2.2.4 定义同步训练 Strategy 及 Optimizer
""""""""""""

在Fleet API中，用户可以使用 ``fleet.DistributedStrategy()`` 接口定义自己想要使用的分布式策略。

其中 ``a_sync`` 选项用于定义参数服务器相关的策略，当其被设定为 ``False`` 时，分布式训练将在同步的模式下进行。反之，当其被设定成 ``True`` 时，分布式训练将在异步的模式下进行。

.. code-block:: python

    # 定义异步训练
    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.a_sync = True

    # 定义同步训练
    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.a_sync = False

    # 定义Geo异步训练, Geo异步目前只支持SGD优化算法
    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.a_sync = True
    dist_strategy.a_sync_configs = {"k_steps": 100}

    optimizer = paddle.optimizer.SGD(learning_rate=0.0001)
    optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
    optimizer.minimize(model.loss)

2.2.5 开始训练
""""""""""""

完成模型及训练策略以后，我们就可以开始训练模型了。因为在参数服务器模式下会有不同的角色，所以根据不同节点分配不同的任务。

对于服务器节点，首先用 ``init_server()`` 接口对其进行初始化，然后启动服务并开始监听由训练节点传来的梯度。

同样对于训练节点，用 ``init_worker()`` 接口进行初始化后， 开始执行训练任务。运行 ``exe.run()`` 接口开始训练，并得到训练中每一步的损失值。

.. code-block:: python

    if fleet.is_server():
        fleet.init_server()
        fleet.run_server()
    else:
        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(paddle.static.default_startup_program())

        fleet.init_worker()

        distributed_training(exe, model)

        fleet.stop_worker()

2.3 运行训练脚本
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

定义完训练脚本后，我们就可以用 ``python3 -m paddle.distributed.launch`` 指令运行分布式任务了。其中 ``server_num`` , ``worker_num`` 分别为服务节点和训练节点的数量。在本例中，服务节点有1个，训练节点有2个。

.. code-block:: bash

    python3 -m paddle.distributed.launch --server_num=1 --worker_num=2 --gpus=0,1 train.py

您将看到显示如下日志信息：

.. code-block:: bash
    
    -----------  Configuration Arguments -----------
    gpus: 0,1
    heter_worker_num: None
    heter_workers:
    http_port: None
    ips: 127.0.0.1
    log_dir: log
    nproc_per_node: None
    server_num: 1
    servers:
    training_script: train.py
    training_script_args: []
    worker_num: 2
    workers:
    ------------------------------------------------
    INFO 2021-05-06 12:14:26,890 launch.py:298] Run parameter-sever mode. pserver arguments:['--worker_num', '--server_num'], cuda count:8
    INFO 2021-05-06 12:14:26,892 launch_utils.py:973] Local server start 1 processes. First process distributed environment info (Only For Debug):
        +=======================================================================================+
        |                        Distributed Envs                      Value                    |
        +---------------------------------------------------------------------------------------+
        |                     PADDLE_TRAINERS_NUM                        2                      |
        |                           TRAINING_ROLE                     PSERVER                   |
        |                                  POD_IP                    127.0.0.1                  |
        |                  PADDLE_GLOO_RENDEZVOUS                        3                      |
        |            PADDLE_PSERVERS_IP_PORT_LIST                 127.0.0.1:34008               |
        |                             PADDLE_PORT                      34008                    |
        |                        PADDLE_WITH_GLOO                        0                      |
        |       PADDLE_HETER_TRAINER_IP_PORT_LIST                                               |
        |                PADDLE_TRAINER_ENDPOINTS         127.0.0.1:18913,127.0.0.1:10025       |
        |               PADDLE_GLOO_HTTP_ENDPOINT                 127.0.0.1:23053               |
        |                     PADDLE_GLOO_FS_PATH                /tmp/tmp8vqb8arq               |
        +=======================================================================================+
    
    INFO 2021-05-06 12:14:26,902 launch_utils.py:1041] Local worker start 2 processes. First process distributed environment info (Only For Debug):
        +=======================================================================================+
        |                        Distributed Envs                      Value                    |
        +---------------------------------------------------------------------------------------+
        |               PADDLE_GLOO_HTTP_ENDPOINT                 127.0.0.1:23053               |
        |                  PADDLE_GLOO_RENDEZVOUS                        3                      |
        |            PADDLE_PSERVERS_IP_PORT_LIST                 127.0.0.1:34008               |
        |                        PADDLE_WITH_GLOO                        0                      |
        |                PADDLE_TRAINER_ENDPOINTS         127.0.0.1:18913,127.0.0.1:10025       |
        |                     FLAGS_selected_gpus                        0                      |
        |                     PADDLE_GLOO_FS_PATH                /tmp/tmp8vqb8arq               |
        |                     PADDLE_TRAINERS_NUM                        2                      |
        |                           TRAINING_ROLE                     TRAINER                   |
        |                     XPU_VISIBLE_DEVICES                        0                      |
        |       PADDLE_HETER_TRAINER_IP_PORT_LIST                                               |
        |                       PADDLE_TRAINER_ID                        0                      |
        |                    CUDA_VISIBLE_DEVICES                        0                      |
        |                     FLAGS_selected_xpus                        0                      |
        +=======================================================================================+
    
    INFO 2021-05-06 12:14:26,921 launch_utils.py:903] Please check servers, workers and heter_worker logs in log/workerlog.*, log/serverlog.* and log/heterlog.*
    INFO 2021-05-06 12:14:33,446 launch_utils.py:914] all workers exit, going to finish parameter server and heter_worker.
    INFO 2021-05-06 12:14:33,446 launch_utils.py:926] all parameter server are killed