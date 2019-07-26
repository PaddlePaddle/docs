..  _cluster_quick_start:

从Paddle Fluid 1.5开始，我们推荐使用Fleet API进行分布式训练，关于Fleet API的介绍可以参考 :ref:`fleet_api`

首先，我们假设读者已经学会单机训练，如果还没单机训练的经验，请参考 :ref:`single_training`
想了解分布式训练，我们可以从单机模拟分布式训练开始，在单台机器上启动多个进程代表多台机器，并进行分布式训练。

为了让读者快速上手，我们采用点击率预估任务作为示例，相关的源码可以参考 xxxx

单机训练代码

.. code:: python

    def train():
        args = parse_args()
        if not os.path.isdir(args.model_output_dir):
            os.mkdir(args.model_output_dir)
    
        dense_input = fluid.layers.data(
               name="dense_input", shape=[dense_feature_dim], dtype='float32')
        sparse_input_ids = [
               fluid.layers.data(name="C" + str(i), shape=[1], lod_level=1, dtype="int64")
               for i in range(1, 27)]
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')

        loss, auc_var, batch_auc_var = ctr_dnn_model_dataset(dense_input, sparse_input_ids, label,
                                                             args.embedding_size, args.sparse_feature_dim)

        optimizer = fluid.optimizer.SGD(learning_rate=1e-4)
        optimizer.minimize(loss)

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())
        dataset = fluid.DatasetFactory().create_dataset()
        dataset.set_use_var([dense_input] + sparse_input_ids + [label])
        pipe_command = "python criteo_reader.py %d" % args.sparse_feature_dim
        dataset.set_pipe_command(pipe_command)
        dataset.set_batch_size(100)
        thread_num = 10
        dataset.set_thread(thread_num)
        whole_filelist = ["raw_data/part-%d" % x for x in range(len(os.listdir("raw_data")))]
        
        epochs = 20
        for i in range(epochs):
            dataset.set_filelist(whole_filelist[:int(0.8*len(whole_filelist))])
            exe.train_from_dataset(program=fluid.default_main_program(),
                                   dataset=dataset,
                                   fetch_list=[auc_var],
                                   fetch_info=["auc"],
                                   debug=False)
            model_dir = args.model_output_dir + '/epoch' + str(i + 1) + ".model"
            sys.stderr.write("epoch%d finished" % (i + 1))
            fluid.io.save_inference_model(model_dir, [dense_input.name] + [x.name for x in sparse_input_ids] + [label.name],                                          [loss, auc_var], exe)

使用FleetAPI进行训练的代码

.. code:: python

    import paddle.fluid.incubate.fleet.base.role_maker as role_maker
    from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet

    def train():
        args = parse_args()
        if not os.path.isdir(args.model_output_dir):
            os.mkdir(args.model_output_dir)
    
        dense_input = fluid.layers.data(
               name="dense_input", shape=[dense_feature_dim], dtype='float32')
        sparse_input_ids = [
               fluid.layers.data(name="C" + str(i), shape=[1], lod_level=1, dtype="int64")
               for i in range(1, 27)]
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')

        loss, auc_var, batch_auc_var = ctr_dnn_model_dataset(dense_input, sparse_input_ids, label,
                                                             args.embedding_size, args.sparse_feature_dim)

        role = role_maker.PaddleCloudRoleMaker()
        exe = fluid.Executor(fluid.CPUPlace())
        fleet.init(role)

        optimizer = fluid.optimizer.SGD(learning_rate=1e-4)
        strategy = DistributeTranspilerConfig()
        strategy.sync_mode = False
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(loss)

        if fleet.is_server():
            fleet.init_server()
            fleet.run_server()
        elif fleet.is_worker():
            fleet.init_worker()
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())
            dataset = fluid.DatasetFactory().create_dataset()
            dataset.set_use_var([dense_input] + sparse_input_ids + [label])
            pipe_command = "python criteo_reader.py %d" % args.sparse_feature_dim
            dataset.set_pipe_command(pipe_command)
            dataset.set_batch_size(100)
            thread_num = 10
            dataset.set_thread(thread_num)
            whole_filelist = ["raw_data/part-%d" % x for x in range(len(os.listdir("raw_data")))]
            
            epochs = 20
            for i in range(epochs):
                dataset.set_filelist(whole_filelist[:int(0.8*len(whole_filelist))])
                exe.train_from_dataset(program=fluid.default_main_program(),
                                       dataset=dataset,
                                       fetch_list=[auc_var],
                                       fetch_info=["auc"],
                                       debug=False)
                
                if fleet.worker_index() == 0:
                    model_dir = args.model_output_dir + '/epoch' + str(i + 1) + ".model"
                    sys.stderr.write("epoch%d finished" % (i + 1))
                    fluid.io.save_inference_model(model_dir, 
                        [dense_input.name] + [x.name for x in sparse_input_ids] + [label.name], [loss, auc_var], exe)
                
                
启动命令

.. code:: python
    
    python -m paddle.distributed.launch_ps --worker_num 2 --server_num 2 dist_train.py

运行日志

如何进行多机分布式训练
请参考百度云运行分布式任务的示例



分布式训练快速开始
==================

准备工作
--------

在本篇文章中，我们将会在介绍如何快速在一个集群中启动一个 PaddlePaddle
的分布式训练任务，在开始之前，请按如下步骤做些准备工作：

1. 准备一个网络连通的训练集群，在本文中我们使用4个训练节点使用 ``*.paddlepaddle.com``
   来表示节点的主机名称，您可以根据实际情况修改它。

2. 在开始之前确保已经阅读过 :ref:`install_steps`
   并且可以在集群的所有节点上可以正常运行 PaddlePaddle。

样例代码
-------

下面使用一个非常简单的线性回归模型作为样例来解释如何启动一个包含2个 ``PSERVER`` 节点以及
2个 ``TRAINER`` 节点的分布式训练任务，您可以将本段代码保存为 ``dist_train.py`` 运行。

.. code:: python

    import os
    import paddle
    import paddle.fluid as fluid

    # train reader
    BATCH_SIZE = 20
    EPOCH_NUM = 30
    BATCH_SIZE = 8

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.uci_housing.train(), buf_size=500),
        batch_size=BATCH_SIZE)

    def train():
        y = fluid.layers.data(name='y', shape=[1], dtype='float32')
        x = fluid.layers.data(name='x', shape=[13], dtype='float32')
        y_predict = fluid.layers.fc(input=x, size=1, act=None)

        loss = fluid.layers.square_error_cost(input=y_predict, label=y)
        avg_loss = fluid.layers.mean(loss)
        opt = fluid.optimizer.SGD(learning_rate=0.001)
        opt.minimize(avg_loss)

        place = fluid.CPUPlace()
        feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
        exe = fluid.Executor(place)

        # fetch distributed training environment setting
        training_role = os.getenv("PADDLE_TRAINING_ROLE", None)
        port = os.getenv("PADDLE_PSERVER_PORT", "6174")
        pserver_ips = os.getenv("PADDLE_PSERVER_IPS", "")
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        eplist = []
        for ip in pserver_ips.split(","):
            eplist.append(':'.join([ip, port]))
        pserver_endpoints = ",".join(eplist)
        trainers = int(os.getenv("PADDLE_TRAINERS"))
        current_endpoint = os.getenv("PADDLE_CURRENT_IP", "") + ":" + port

        t = fluid.DistributeTranspiler()
        t.transpile(
            trainer_id = trainer_id,
            pservers = pserver_endpoints,
            trainers = trainers)

        if training_role == "PSERVER":
            pserver_prog = t.get_pserver_program(current_endpoint)
            startup_prog = t.get_startup_program(current_endpoint, pserver_prog)
            exe.run(startup_prog)
            exe.run(pserver_prog)
        elif training_role == "TRAINER":
            trainer_prog = t.get_trainer_program()
            exe.run(fluid.default_startup_program())

            for epoch in range(EPOCH_NUM):
                for batch_id, batch_data in enumerate(train_reader()):
                    avg_loss_value, = exe.run(trainer_prog,
                                          feed=feeder.feed(batch_data),
                                          fetch_list=[avg_loss])
                    if (batch_id + 1) % 10 == 0:
                        print("Epoch: {0}, Batch: {1}, loss: {2}".format(
                            epoch, batch_id, avg_loss_value[0]))
            # destory the resource of current trainer node in pserver server node
            exe.close()
        else:
            raise AssertionError("PADDLE_TRAINING_ROLE should be one of [TRAINER, PSERVER]")

    train()

环境变量说明
-----------

在启动分布式训练任务时，使用不同的环境变量来表示不同的节点角色，具体如下：

.. list-table::
  :header-rows: 1

  * - 环境变量
    - 数据类型
    - 样例
    - 描述
  * - :code:`PADDLE_TRAINING_ROLE`
    - str
    - :code:`PSERVER,TRAINER`
    - 当前训练节点角色
  * - :code:`PADDLE_PSERVER_IPS`
    - str
    - :code:`ps0.paddlepaddle.com,ps1.paddlepaddle.com`
    - 分布式训练任务中所有 PSERVER 节点的 IP 地址或 hostname, 使用","分隔
  * - :code:`PADDLE_PSERVER_PORT`
    - int
    - 6174
    - PSERVER 进程监听的端口
  * - :code:`PADDLE_TRAINERS`
    - int
    - 2
    - 分布式训练任务中 trainer 节点的数量
  * - :code:`PADDLE_CURRENT_IP`
    - str
    - :code:`ps0.paddlepaddle.com`
    - 当前 PSERVER 节点的 IP 地址或 hostname
  * - :code:`PADDLE_TRAINER_ID`
    - str 
    - 0
    - 当前 TRAINER 节点的 ID (唯一)， 取值范围为 [0, PADDLE_TRAINERS)

注： 环境变量只是获取运行时信息的一种方式，实际任务中可以采用命令行参数等方式获取运行时信息。

分布式训练相关 API
------------------

DistributeTranspiler
~~~~~~~~~~~~~~~~~~~~~~

基于 pserver-trainer 架构的的分布式训练任务分为两种角色： Parameter Server(PSERVER) 以及 TRAINER, 
在 Fluid 中，用户只需配置单机训练所需要的网络配置, ``DistributeTranspiler`` 模块会自动地根据
当前训练节点的角色将用户配置的单机网路配置改写成 PSERVER 和 TRAINER 需要运行的网络配置:

.. code:: python

    t = fluid.DistributeTranspiler()
    t.transpile(
        trainer_id = trainer_id,                   
        pservers = pserver_endpoints,    
        trainers = trainers)
    if PADDLE_TRAINING_ROLE == "TRAINER":
        # fetch the trainer program and execute it
        trainer_prog = t.get_trainer_program()
        ...

    elif PADDLE_TRAINER_ROLE == "PSERVER":
        # fetch the pserver program and execute it
        pserver_prog = t.get_pserver_program(current_endpoint) 
        ...

exe.close()
~~~~~~~~~~~~~~

PSERVER 节点中会保存所有 TRAINER 节点的状态信息，在 TRAINER 结束训练时需要调用 ``exe.close()``
通知所有 PSERVER 节点释放当前 TRAINER 节点的资源:

.. code:: python

    exe = fluid.Executor(fluid.CPUPlace())
    # training process ...
    exe.close() # notify PServer to destory the resource

注意：所有的trainer在退出时都需要调用exe.close()。


启动分布式训练任务
--------------------

.. list-table::
   :header-rows: 1

   * - 启动节点
     - 启动命令
     - 说明
   * - ps0.paddlepaddle.com
     - :code:`PADDLE_TRAINING_ROLE=PSERVER PADDLE_CURRENT_IP=ps0.paddlepaddle.com PADDLE_PSERVER_IPS=ps0.paddlepaddle.com,ps1.paddlepaddle.com PADDLE_TRAINERS=2 PADDLE_PSERVER_PORT=6174 python fluid_dist.py`
     - 启动 PSERVER 节点
   * - ps1.paddlepaddle.com
     - :code:`PADDLE_TRAINING_ROLE=PSERVER PADDLE_CURRENT_IP=ps1.paddlepaddle.com PADDLE_PSERVER_IPS=ps0.paddlepaddle.com,ps1.paddlepaddle.com PADDLE_TRAINERS=2 PADDLE_PSERVER_PORT=6174 python fluid_dist.py`
     - 启动 PSERVER 节点
   * - trainer0.paddlepaddle.com
     - :code:`PADDLE_TRAINING_ROLE=TRAINER PADDLE_PSERVER_IPS=ps0.paddlepaddle.com,ps1.paddlepaddle.com PADDLE_TRAINERS=2 PADDLE_TRAINER_ID=0 PADDLE_PSERVER_PORT=6174 python fluid_dist.py`
     - 启动第0号 TRAINER 节点
   * - trainer1.paddlepaddle.com
     - :code:`PADDLE_TRAINING_ROLE=TRAINER PADDLE_PSERVER_IPS=ps0.paddlepaddle.com,ps1.paddlepaddle.com PADDLE_TRAINERS=2 PADDLE_TRAINER_ID=1 PADDLE_PSERVER_PORT=6174 python fluid_dist.py`
     - 启动第1号 TRAINER 节点
