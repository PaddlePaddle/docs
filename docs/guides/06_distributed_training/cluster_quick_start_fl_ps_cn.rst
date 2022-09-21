
..  _cluster_quick_start_fl_ps:

快速开始-联邦参数服务器
-------------------------

* 在传统的搜索、推荐场景中，为了训练一个全局的模型，通常需要收集所有用户的原始数据并上传至服务端，这样的做法往往存在用户隐私泄露问题。
* 联邦学习使得模型训练的整个过程中，用户的原始数据始终保留在用户（Client）本地，服务端（Server）和用户之间通过共享加密的或不包含隐私信息的中间参数的方式，进行模型训练和参数更新，进而在保护用户隐私的前提下构建一个有效的机器学习模型
* 推荐模型在联邦场景下如何更快、更好地进行训练越来越受到工业界和学术界的关注。

本示例将向你展示百度飞桨联邦参数服务器（FL-PS）的能力、教你如何使用它以及在它基础上做二次开发。

1.1 任务介绍
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

本节将采用推荐领域非常经典的模型 NCF 为例，介绍如何使用飞桨分布式完成 FL-PS 训练任务。

FL-PS 训练基于飞桨静态图，在这之前，请用户了解下 NCF 模型的单机静态图训练示例以及本地单机模拟分布式的使用方法：\ `https://github.com/PaddlePaddle/PaddleRec/tree/master/models/recall/ncf`_\。

在传统 PS 基础上，通过生成异构数据集、开启中心调度功能（Coordinator）进行 Client 选择、自定义配置 Client 端私有稀疏参数和 Server 端公共稀疏参数等手段，提升 FL-PS 的训练精度和效率。

更多使用细节请阅读 \FL-PS 帮助文档：`https://github.com/PaddlePaddle/PaddleRec/blob/master/models/recall/ncf/fl_ps_help.md`_\.

本功能依赖 PaddlePaddle2.4 及以上版本的飞桨开源框架，或者用户从 PaddlePaddle develop 分支进行源码编译。

1.2 操作方法
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

FL-PS 训练主要包括如下几个部分：

    1. 准备样本，构建 dataset
    2. 准备配置文件，设定分布式策略
    3. 加载模型
    4. 开始训练

用户可从 FL-PS 训练脚本 `https://github.com/PaddlePaddle/PaddleRec/blob/master/tools/static_fl_trainer.py` 入手梳理详细流程。

1.2.1 准备样本
""""""""""""

* 在 PaddleRec/datasets/movielens_pinterest_NCF 目录中执行: sh run.sh，获取初步处理过的训练数据（big_train）和测试数据（test_data）
* 从 MovieLens 官网 `https://grouplens.org/datasets/movielens/1m/` 下载 ml-1m 数据集，获取 users.dat 文件（可自定义存储路径，但需要和 gen_heter_data.py 脚本中路径保持一致），后续用于构造异构数据集（按 zipcode 的首位数字划分）
* 在 PaddleRec/datasets/movielens_pinterest_NCF/fl_data 中新建目录 fl_test_data 和 fl_train_data，用于存放每个 client 上的训练数据集和测试数据集
* 在 PaddleRec/datasets/movielens_pinterest_NCF/fl_data 目录中执行: python gen_heter_data.py，生成 10 份训练数据
    * 总样本数 4970844（按 1:4 补充负样本）：0 - 518095，1 - 520165，2 - 373605，3 - 315550，4 - 483779，5 - 495635，6 - 402810，7 - 354590，8 - 262710，9 - 1243905
    * 样本数据每一行表示：物品 id，用户 id，标签

.. code-block:: python

    def init_reader(self):
        self.train_dataset, self.train_file_list = get_reader(self.input_data,
                                                              config)
        self.test_dataset, self.test_file_list = get_infer_reader(
            self.input_data, config)

        if self.role is not None:
            self.fl_client = MyFLClient()
            self.fl_client.set_basic_config(self.role, self.config,
                                            self.metrics)
        else:
            raise ValueError("self.role is none")

        self.fl_client.set_train_dataset_info(self.train_dataset,
                                              self.train_file_list)
        self.fl_client.set_test_dataset_info(self.test_dataset,
                                             self.test_file_list)

        example_nums = 0
        self.count_method = self.config.get("runner.example_count_method",
                                            "example")
        if self.count_method == "example":
            example_nums = get_example_num(self.train_file_list)
        elif self.count_method == "word":
            example_nums = get_word_num(self.train_file_list)
        else:
            raise ValueError(
                "Set static_benchmark.example_count_method for example / word for example count."
            )
        self.fl_client.set_train_example_num(example_nums)

1.2.2 配置文件
""""""""""""

所有的配置参数均写在文件 `config_fl.yaml` 里：

.. code-block:: python

    runner:
        sync_mode: "geo" # 可选, string: sync/async/geo
        #with_coodinator: 1 # 1 表示开启中心调度功能
        geo_step: 100 # 必选, int, 在 geo 模式下控制本地的迭代次数
        split_file_list: True # 可选, bool, 若每个节点上都拥有全量数据，则需设置为 True
        thread_num: 1 # 多线程配置

        # reader 类型，分布式下推荐 QueueDataset
        reader_type: "QueueDataset" # DataLoader / QueueDataset / RecDataset
        pipe_command: "python queuedataset_reader.py" # QueueDataset 模式下的数据 pipe 命令
        dataset_debug: False # QueueDataset 模式下 Profiler 开关

        train_data_dir: "../../../datasets/movielens_pinterest_NCF/fl_data/fl_train_data"
        train_reader_path: "movielens_reader"  # importlib format
        train_batch_size: 512
        model_save_path: "output_model_ncf"

        use_gpu: False
        epochs: 2
        print_interval: 50

        test_data_dir: "../../../datasets/movielens_pinterest_NCF/fl_data/fl_test_data"
        infer_reader_path: "movielens_reader"  # importlib format
        infer_batch_size: 1
        infer_load_path: "output_model_ncf"
        infer_start_epoch: 2
        infer_end_epoch: 3

        need_dump: True
        dump_fields_path: "/home/wangbin/the_one_ps/ziyoujiyi_PaddleRec/PaddleRec/models/recall/ncf"
        dump_fields: ['item_input', 'user_input']
        dump_param: []
        local_sparse: ['embedding_0.w_0']
        remote_sparse: ['embedding_1.w_0']

    hyper_parameters:
        optimizer:
            class: adam
            learning_rate: 0.001
        num_users: 6040
        num_items: 3706
        mf_dim: 8
        mode: "NCF_MLP"  # optional: NCF_NeuMF, NCF_GMF, NCF_MLP
        fc_layers: [64, 32, 16, 8]

1.2.3 加载模型
""""""""""""

.. code-block:: python

    def init_network(self):
        self.model = get_model(self.config)
        self.input_data = self.model.create_feeds()
        self.metrics = self.model.net(self.input_data)
        self.model.create_optimizer(get_strategy(self.config))  ## get_strategy
        if self.pure_bf16:
            self.model.optimizer.amp_init(self.place)

1.2.4 开始训练
""""""""""""

有了训练脚本后，我们就可以用 ``fleetrun`` 指令运行分布式任务了。 ``fleetrun`` 是飞桨封装的分布式启动命令，命令参数 ``server_num`` , ``worker_num``,  ``coordinator_num``分别为服务节点、训练节点、中心调度节点的数量（目前只支持一个 Coordinator 节点）。在本例中，服务节点有 1 个，训练节点有 10 个。运行训练脚本之前，请确保所使用的端口没有被占用

接着，进入 PaddleRec 目录：PaddleRec/models/recall/ncf，

1. 使用 Coordinator 功能

* 首先将 config_fl.yaml 中的参数 ``local_sparse`` 和 ``remote_sparse`` 配置注释掉，参数 ``with_coodinator`` 置为 1

.. code-block:: bash

    fleetrun --worker_num=10 --workers="127.0.0.1:9000,127.0.0.1:9001,127.0.0.1:9002,127.0.0.1:9003,127.0.0.1:9004,127.0.0.1:9005,127.0.0.1:9006,127.0.0.1:9007,127.0.0.1:9008,127.0.0.1:9009" --server_num=1 --servers="127.0.0.1:10000" --coordinator_num=1 --coordinators="127.0.0.1:10001" ../../../tools/static_fl_trainer.py -m config_fl.yaml

* 详细运行日志信息保存在 log/workerlog.*, log/serverlog.* , log/coordinatorlog.* 里，以下是运行成功时 coordinator 进程打印的部分信息：

.. code-block:: bash

    >>> all trainer endpoints: ['127.0.0.1:9000', '127.0.0.1:9001', '127.0.0.1:9002', '127.0.0.1:9003', '127.0.0.1:9004', '127.0.0.1:9005', '127.0.0.1:9006', '127.0.0.1:9007', '127.0.0.1:9008', '127.0.0.1:9009']
    I0921 10:29:35.962728 45248 coordinator_client.cc:123] fl-ps > coordinator connect to fl_client: 0
    I0921 10:29:35.962761 45248 coordinator_client.cc:123] fl-ps > coordinator connect to fl_client: 1
    I0921 10:29:35.962771 45248 coordinator_client.cc:123] fl-ps > coordinator connect to fl_client: 2
    I0921 10:29:35.962779 45248 coordinator_client.cc:123] fl-ps > coordinator connect to fl_client: 3
    I0921 10:29:35.962786 45248 coordinator_client.cc:123] fl-ps > coordinator connect to fl_client: 4
    I0921 10:29:35.962792 45248 coordinator_client.cc:123] fl-ps > coordinator connect to fl_client: 5
    I0921 10:29:35.962797 45248 coordinator_client.cc:123] fl-ps > coordinator connect to fl_client: 6
    I0921 10:29:35.962802 45248 coordinator_client.cc:123] fl-ps > coordinator connect to fl_client: 7
    I0921 10:29:35.962810 45248 coordinator_client.cc:123] fl-ps > coordinator connect to fl_client: 8
    I0921 10:29:35.962815 45248 coordinator_client.cc:123] fl-ps > coordinator connect to fl_client: 9
    I0921 10:29:35.962828 45248 communicator.cc:1536] fl-ps > StartCoordinatorClient finish!
    I0921 10:29:35.965075 45248 server.cpp:1066] Server[paddle::distributed::CoordinatorService] is serving on port=10001.
    I0921 10:29:35.965721 45248 coordinator_client.cc:167] fl-ps > coordinator service addr: 127.0.0.1, 10001, 0
    I0921 10:29:35.965732 45248 communicator.cc:1547] fl-ps > StartCoordinatorServer finished!
    2022-09-21 10:29:35,965 INFO [coordinator.py:344] fl-ps > running make_fl_strategy(loop) in coordinator

    2022-09-21 10:29:35,965 - INFO - fl-ps > running make_fl_strategy(loop) in coordinator

    I0921 10:29:55.610915 45534 coordinator_client.cc:45] fl-ps > recv from client id: 9, msg_type: 200
    I0921 10:29:55.610915 45540 coordinator_client.cc:45] fl-ps > recv from client id: 5, msg_type: 200
    I0921 10:29:55.610915 45539 coordinator_client.cc:45] fl-ps > recv from client id: 7, msg_type: 200
    I0921 10:29:55.610915 45538 coordinator_client.cc:45] fl-ps > recv from client id: 8, msg_type: 200
    I0921 10:29:55.610915 45533 coordinator_client.cc:45] fl-ps > recv from client id: 2, msg_type: 200

2. 使用稀疏参数切分功能

* 首先将 config_fl.yaml 中的 ``with_coodinator`` 注释掉，放开参数 ``local_sparse`` 和 ``remote_sparse`` 配置

.. code-block:: bash

    fleetrun --worker_num=10 --workers="127.0.0.1:9000,127.0.0.1:9001,127.0.0.1:9002,127.0.0.1:9003,127.0.0.1:9004,127.0.0.1:9005,127.0.0.1:9006,127.0.0.1:9007,127.0.0.1:9008,127.0.0.1:9009" --server_num=1 --servers="127.0.0.1:10000" ../../../tools/static_fl_trainer.py -m config_fl.yaml

* 详细运行日志信息保存在 log/workerlog.*, log/serverlog.* 里，以下是运行成功时 worker 进程打印的部分信息：

.. code-block:: bash

    time: [2022-09-21 09:58:58], batch: [50], Epoch 0 Var Loss[1]:[0.609486], Epoch 0 Var Auc[1]:[0.500178]
    time: [2022-09-21 09:58:58], batch: [100], Epoch 0 Var Loss[1]:[0.501269], Epoch 0 Var Auc[1]:[0.500078]
    time: [2022-09-21 09:58:58], batch: [150], Epoch 0 Var Loss[1]:[0.49927], Epoch 0 Var Auc[1]:[0.500261]
    time: [2022-09-21 09:58:59], batch: [200], Epoch 0 Var Loss[1]:[0.498443], Epoch 0 Var Auc[1]:[0.501497]
    time: [2022-09-21 09:58:59], batch: [250], Epoch 0 Var Loss[1]:[0.499356], Epoch 0 Var Auc[1]:[0.501259]
    time: [2022-09-21 09:58:59], batch: [300], Epoch 0 Var Loss[1]:[0.498732], Epoch 0 Var Auc[1]:[0.502684]
    time: [2022-09-21 09:59:00], batch: [350], Epoch 0 Var Loss[1]:[0.500202], Epoch 0 Var Auc[1]:[0.50294]
    time: [2022-09-21 09:59:00], batch: [400], Epoch 0 Var Loss[1]:[0.498004], Epoch 0 Var Auc[1]:[0.504768]
    time: [2022-09-21 09:59:01], batch: [450], Epoch 0 Var Loss[1]:[0.498487], Epoch 0 Var Auc[1]:[0.504689]


1.3 二次开发
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

用户可以基于 Paddle develop 分支进行 FL-PS 的二次开发：

1.3.1 编译安装
""""""""""""

.. code-block:: bash

    1）去 https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/compile/linux-compile.html 找到 develop/Linux/源码编译/CPU/ 的开发镜像，在 docker 中开发
    2）在 Paddle 根目录下，新建 build 目录
    3）cd build
    4）cmake .. -DPY_VERSION=3.7 -DWITH_GPU=OFF -DCMAKE_BUILD_TYPE=Release -DWITH_DISTRIBUTE=ON -DWITH_PSCORE=ON -WITH_AVX=OFF -DWITH_TESTING=OFF -DWITH_FLPS=ON
    5) make -j
    6）python -m pip install python/dist/paddlepaddle-0.0.0-cp37-cp37m-linux_x86_64.whl -U


1.3.2 Coordinator 模块
""""""""""""

用户可以基于文件 `Paddle/python/paddle/distributed/ps/coordinator.py` 中定义的相关基类进行继承开发，用户自定义的各种 Client 选择算法均可以用 python 代码实现，从类 `ClientSelectorBase` 继承。

1.3.3 构造自定义异构数据集
""""""""""""

参考脚本 `gen_heter_data.py` 写法。


备注：本教程主要介绍了横向联邦 PS 的使用方法，关于纵向联邦 PS 的使用，请参考\ `https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/ps/test_fl_ps.py`_\，使用 1.3.1 节的编译命令，再执行下述命令即可

.. code-block:: bash
    ctest -R test_fl_ps -V

由于该单测需要从网上下载数据集，运行时请确保数据成功下载下来。
