
..  _cluster_overview_ps:

参数服务器概述
-------------------------

通常来讲，一个推荐系统的全流程可以用下图来简单示例：

.. image:: ./images/whole_process.png
  :width: 800
  :alt: whole_process
  :align: center

1. 推荐系统线上服务产生的日志，经过特征拼接后落盘到存储介质上形成数据源，数据经过处理进入分布式训练系统。
2. 训练过程持续产生用于线上推理服务的向量库及模型。
3. 训练产生的向量灌入在线服务向量库，模型配送到线上推理服务进行更新，在线推理服务开始使用更新后的模型进行推理。

相较于CV、NLP等场景，推荐/搜索场景下的训练的有以下几大特点：

1. 稀疏参数量大：推荐场景下的特征中包含大量的id类特征（例如userid、itemid），这些id类特征会对应大量的embedding（称为稀疏参数），通常参数量在百亿级别及以上，且随训练过程不断增加。
2. 训练数据量大：线上推理服务会源源不断产生训练数据进入分布式训练中，训练数据量级巨大，单机训练速度过慢。
3. 流式训练：数据不是一次性放入训练系统中，而是随着时间流式地加入到训练过程中去，并实时产生模型配送到线上推理服务中，因此对训练时间有严格要求。

由于训练数据量大且对训练时间的严格要求，单机训练很难满足需求，因此需要引入多个训练节点，以数据并行的方式进行模型训练；

同时，为了能够存储海量的稀疏参数并可以让多个训练节点共享参数，引入服务节点对模型参数进行中心化管理（存储和更新），当单个服务节点的存储空间不足或者训练节点个数太多导致服务节点成为瓶颈时，可以引入多个服务节点。

这就是分布式训练中的参数服务器模式。

.. image:: ./images/ps.JPG
  :width: 600
  :alt: ps
  :align: center

上图即为参数服务器的示意图，该模式下的节点/进程有两种不同的角色：

1. 训练节点（Trainer/Worker）：负责训练，完成数据读取、从服务节点拉取参数、前向计算、反向梯度计算等过程，并将计算出的梯度上传至服务节点。
2. 服务节点（Server）：负责模型参数的集中式存储和更新，当服务节点接收到来自训练节点的参数梯度时，服务节点会将梯度聚合并更新参数，供训练节点拉取进行下一轮的训练。

参数服务器模式对于存储超大规模模型参数的训练场景十分友好，常被用于训练拥有海量稀疏参数的搜索推荐领域模型。

1 技术选型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

随着硬件种类越来越多，推荐/搜索场景下的模型越来越复杂，参数服务器也产生了多种类型，需结合业务模型特点和硬件成本，选择最优的参数服务器方案。

1.1 纯CPU参数服务器
""""""""""""

纯CPU参数服务器（CPUPS）采用多台硬件型号完全一致的CPU机器进行训练，由高性能异步训练Worker、高效通信策略和高性能Server组成。

由于使用的CPU数量较多，训练中能够充分展示CPU多核的吞吐量优势，在异步训练模式下训练简单模型可以极大提升数据吞吐量。

纯CPU参数服务器相关原理可以参考：\ `CPUPS原理 <https://>`_\

1.2 纯GPU参数服务器
""""""""""""

随着模型网络越来越复杂，对算力要求越来越高，在数据量不变的情况下，CPU计算性能差的弱势就会显现。虽然可以通过增加 CPU 机器数量来解决，甚至可以增加上百台，但是这种方法不仅成本大幅提高，而且集群的稳定性和扩展性也存在较大的问题。

因此纯GPU参数服务器（GPUPS）应运而生，通常100台CPU机器才能训练的模型，仅需1台多卡 GPU 机器即可完成训练。

纯GPU参数服务器相关原理可以参考：\ `GPUPS原理 <https://>`_\

1.3 异构参数服务器
""""""""""""

为进一步提升训练资源利用率，解除训练节点必须严格使用同一种硬件型号的枷锁，提出了通用异构参数服务器（HeterPS）。

HeterPS使训练任务对硬件型号不敏感，即可以同时使用不同的硬件混合异构训练，如 CPU、AI专用芯片（如百度昆仑XPU）以及不同型号的GPU，如 v100、P40、K40 等。同时还可以解决大规模稀疏特征模型训练场景下 IO 占比过高导致的芯片资源利用率过低的问题。

异构参数服务器的最大亮点是硬件感知的任务切分。将IO密集型任务（如数据读取、Embedding查询）切分给CPU机器，将计算密集型任务切分给GPU机器；用户可以根据子任务的计算复杂度来灵活决定机器配比，并且还可以兼容传统纯CPU参数服务器和纯GPU参数服务器所支持的训练任务。

通用异构参数服务器相关原理可以参考：\ `HeterPS原理 <https://>`_\


2 使用方法
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

使用参数服务器的一个简单的代码示例如下：

.. code-block:: python

    import paddle
    # 导入分布式训练需要的依赖fleet
    import paddle.distributed.fleet as fleet
    # 导入模型
    from model import WideDeepModel

    # 参数服务器目前只支持静态图，需要使用enable_static()
    paddle.enable_static()

    # 加载模型并构造优化器
    model = WideDeepModel()
    model.net(is_train=True)
    optimizer = paddle.optimizer.SGD(learning_rate=0.0001)

    # 初始化fleet
    fleet.init(is_collective=False)
    # 设置分布式策略（异步更新方式）
    strategy = fleet.DistributedStrategy()
    strategy.a_sync = True

    # 构造分布式优化器
    optimizer = fleet.distributed_optimizer(optimizer, strategy)
    optimizer.minimize(model.cost)

    if fleet.is_server():
        # 初始化服务节点
        fleet.init_server()
        # 启动服务节点，即可接收来自训练节点的请求
        fleet.run_server()

    if fleet.is_worker():
        # 训练节点的具体训练过程
        ...
        # 训练结束终止训练节点
        fleet.stop_worker()

其中示例代码中省略的，训练节点的一个完整的训练过程应该包含以下几个部分：

    1. 获取之前训练已经保存好的模型，并加载模型（如果之前没有保存模型，则跳过加载模型这一步）。
    2. 分Pass训练，在每一个Pass的训练过程中，分为如下几步：
      a. 加载数据。
      b. 分布式训练并获取训练指标（AUC等）。
      c. 分布式预测：主要用于召回模块的离线建库部分。
    3. 保存模型：
      a. Checkpoint Model：用于下次训练开始时的模型加载部分。
      b. Inference Model：用于线上推理部署。
    
完整训练示例代码请参考：\ `CPUPS示例 <https://>`_\、\ `GPUPS示例 <https://>`_\，本节只介绍飞桨参数服务器在训练过程中需要使用到的与单机不同的API。

2.1 大规模稀疏参数
""""""""""""

为存储海量的稀疏参数，参数服务器使用 ``paddle.static.nn.sparse_embedding()`` 取代 ``paddle.static.nn.embedding`` 作为embedding lookup层的算子。

``paddle.static.nn.sparse_embedding()`` 采用稀疏模式进行梯度的计算和更新，输入接受[0, UINT64]范围内的特征ID，支持稀疏参数各种高阶配置（特征准入、退场等），更加符合流式训练的功能需求。

.. code-block:: python

    import paddle

    # sparse_embedding输入接受[0, UINT64]范围内的特征ID，参数size的第一维词表大小无用，可指定任意整数
    # 大规模稀疏场景下，参数规模初始为0，会随着训练的进行逐步扩展
    sparse_feature_num = 10
    embedding_size = 64

    input = paddle.static.data(name='ins', shape=[1], dtype='int64')

    emb = paddle.static.nn.sparse_embedding((
        input=input,
        size=[sparse_feature_num, embedding_size],
        param_attr=paddle.ParamAttr(name="SparseFeatFactors",
        initializer=paddle.nn.initializer.Uniform()))

2.2 加载数据
""""""""""""

由于搜索推荐场景涉及到的训练数据通常较大，为提升训练中的数据读取效率，参数服务器采用Dataset进行高性能的IO。

Dataset是为多线程及全异步方式量身打造的数据读取方式，每个数据读取线程会与一个训练线程耦合，形成了多生产者-多消费者的模式，会极大的加速模型训练过程。

.. image:: ./images/dataset.JPG
  :width: 600
  :alt: dataset
  :align: center

Dataset有两种不同的类型：
1. QueueDataset：随训练流式读取数据。
2. InmemoryDataset：训练数据全部读入训练节点内存，然后分配至各个训练线程，支持全局秒级打散数据（global_shuffle）。

.. code-block:: python

    dataset = paddle.distributed.QueueDataset()
    thread_num = 1
    
    # use_var指定网络中的输入数据，pipe_command指定数据处理脚本
    # 要求use_var中输入数据的顺序与数据处理脚本输出的特征顺序一一对应
    dataset.init(use_var=model.inputs, 
                 pipe_command="python reader.py", 
                 batch_size=batch_size, 
                 thread_num=thread_num)

    train_files_list = [os.path.join(train_data_path, x)
                        for x in os.listdir(train_data_path)]
    
    # set_filelist指定dataset读取的训练文件的列表
    dataset.set_filelist(train_files_list)

更多dataset用法参见\ `使用InMemoryDataset/QueueDataset进行训练 <https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/parameter_server/performance/dataset.html>`_\。

2.3 分布式训练及预测
""""""""""""

与数据加载dataset相对应的，使用 ``exe.train_from_dataset()`` 接口进行分布式训练。

.. code-block:: python
    exe.train_from_dataset(paddle.static.default_main_program(),
                          dataset,
                          paddle.static.global_scope(), 
                          debug=False, 
                          fetch_list=[model.cost],
                          fetch_info=["loss"],
                          print_period=1)

分布式预测使用 ``exe.infer_from_dataset()`` 接口，与分布式训练的区别是，预测阶段训练节点不向服务节点发送梯度。

.. code-block:: python
    exe.infer_from_dataset(paddle.static.default_main_program(),
                          dataset,
                          paddle.static.global_scope(), 
                          debug=False, 
                          fetch_list=[model.cost],
                          fetch_info=["loss"],
                          print_period=1)

2.4 分布式指标计算
""""""""""""

分布式指标是指在分布式训练任务中用以评测模型效果的指标。
由于参数服务器存在多个训练节点，传统的指标计算只能评测当前节点的数据，而分布式指标需要汇总所有节点的全量数据，进行全局指标计算。

分布式指标计算的接口位于 ``paddle.distributed.fleet.metrics`` ，其中封装了包括AUC、Accuracy、MSE等常见指标计算。

以AUC指标为例，全局AUC指标计算示例如下：

.. code-block:: python
    # 组网阶段，AUC算子在计算auc指标同时，返回正负样例中间统计结果（stat_pos, stat_neg）
    auc, batch_auc, [batch_stat_pos, batch_stat_neg, stat_pos, stat_neg] = \
        paddle.static.auc(input=pred, label=label)

    # 利用AUC算子返回的中间计算结果，以及fleet提供的分布式指标计算接口，完成全局AUC计算。
    global_auc = fleet.metrics.auc(stat_pos, stat_neg)

更多分布式指标用法参见\ `分布式指标计算 <https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/parameter_server/ps_distributed_metrics.html>`_\。


2.5 模型保存与加载
""""""""""""

save_persistables
save_inference_model
load_model

3 进阶教程
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. GPUPS示例
2. HeterPS示例
3. 稀疏参数配置（accessor）
4. 二次开发