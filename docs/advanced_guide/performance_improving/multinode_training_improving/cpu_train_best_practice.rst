.. _api_guide_cpu_training_best_practice:

####################
分布式 CPU 训练优秀实践
####################

提高 CPU 分布式训练的训练速度，主要要从四个方面来考虑：
1）提高训练速度，主要是提高 CPU 的使用率；2）提高通信速度，主要是减少通信传输的数据量；3）提高数据 IO 速度；4）更换分布式训练策略，提高分布式训练速度。

提高 CPU 的使用率
=============

提高 CPU 使用率主要依赖 :code:`ParallelExecutor`，可以充分利用多个 CPU 的计算能力来加速计算。

简单实例用法：

.. code-block:: python

    # 配置执行策略，主要是设置线程数
    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = 8

    # 配置构图策略，对于 CPU 训练而言，应该使用 Reduce 模式进行训练
    build_strategy = fluid.BuildStrategy()
    if int(os.getenv("CPU_NUM")) > 1:
        build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce

    pe = fluid.ParallelExecutor(
        use_cuda=False,
        loss_name=avg_cost.name,
        main_program=main_program,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)

以上参数中：

- :code:`num_threads` ： 模型训练使用的线程数，最好和训练所在机器的物理 CPU 核数接近
- :code:`reduce_strategy` ： 对于 CPU 训练而言，应该选择 fluid.BuildStrategy.ReduceStrategy.Reduce


通用环境变量配置：

- :code:`CPU_NUM` ：模型副本 replica 的个数，最好和 num_threads 一致


提高通信速度
==========

要减少通信数据量，提高通信速度，主要是使用稀疏更新 ，目前支持  :ref:`api_guide_sparse_update` 的主要是  :ref:`cn_api_fluid_layers_embedding` 。

.. code-block:: python

    data = fluid.layers.data(name='ids', shape=[1], dtype='int64')
    fc = fluid.layers.embedding(input=data, size=[dict_size, 16], is_sparse=True)

以上参数中：

- :code:`is_sparse` ： 配置 embedding 使用稀疏更新，如果 embedding 的 dict_size 很大，而每次数据 data 很少，建议使用 sparse 更新方式。


提高数据 IO 速度
==========

要提高 CPU 分布式的数据 IO 速度，可以首先考虑使用 dataset API 进行数据读取。 dataset 是一种多生产者多消费者模式的数据读取方法，默认情况下耦合数据读取线程与训练线程，在多线程的训练中，dataset 表现出极高的性能优势。

API 接口介绍可以参考： :ref:`cn_api_distributed_QueueDataset`

结合实际的网络，比如 CTR-DNN 模型，引入的方法可以参考：https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleRec/ctr/dnn

最后使用 :code:`train_from_dataset` 接口来进行网络的训练：

.. code-block:: python

    dataset = fluid.DatasetFactory().create_dataset()
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    exe.train_from_dataset(program=fluid.default_main_program(),dataset=dataset)


更换分布式训练策略
==========

CPU 分布式训练速度进一步提高的核心在于选择合适的分布式训练策略，比如定义通信策略、编译策略、执行策略等等。paddlepaddle 于 v1.7 版本发布了 :code:`DistributedStrategy` 功能，可以十分灵活且方便的指定分布式运行策略。

首先需要在代码中引入相关库：

.. code-block:: python

    from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
    import paddle.fluid.incubate.fleet.base.role_maker as role_maker
    from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler.distributed_strategy_factory import DistributedStrategyFactory

然后指定 CPU 分布式运行的训练策略，目前可选配置有四种：同步训练（Sync）、异步训练（Async）、半异步训练（Half-Async）以及 GEO 训练。


通过如下代码引入上述策略的默认配置，并进行 CPU 分布式训练：

.. code-block:: python

    # step1: 引入 CPU 分布式训练策略
    # 同步训练策略
    strategy = DistributedStrategyFactory.create_sync_strategy()
    # 半异步训练策略
    strategy = DistributedStrategyFactory.create_half_async_strategy()
    # 异步训练策略
    strategy = DistributedStrategyFactory.create_async_strategy()
    # GEO 训练策略
    strategy = DistributedStrategyFactory.create_geo_strategy(update_frequency=400)

    # step2: 定义节点角色
    role = role_maker.PaddleCloudRoleMaker()
    fleet.init(role)

    # step3: 分布式训练 program 构建
    optimizer = fluid.optimizer.SGD(learning_rate) # 以 SGD 优化器为例
    optimizer = fleet.distributed_optimizer(optimizer, strategy)
    optimizer.minimize(loss)

    # step4.1: 启动参数服务器节点（Server）
    if fleet.is_server():
        fleet.init_server()
        fleet.run_server()

    # step4.2: 启动训练节点（Trainer）
    elif fleet.is_worker():
        fleet.init_worker()
        exe.run(fleet.startup_program)
        # Do training
        exe.run(fleet.main_program)
        fleet.stop_worker()


paddlepaddle 支持对训练策略中的细节进行调整：

- 创建 compiled_program 所需的 build_strategy 及 exec_strategy 可以直接基于 strategy 获得

.. code-block:: python

    compiled_program = fluid.compiler.CompiledProgram(fleet.main_program).with_data_parallel(
                                                                            loss_name=loss.name,
                                                                            build_strategy=strategy.get_build_strategy(),
                                                                            exec_strategy=strategy.get_execute_strategy())


- 自定义训练策略细节，支持对 DistributeTranspilerConfig、TrainerRuntimeConfig、ServerRuntimeConfig、fluid.ExecutionStrategy、fluid.BuildStrategy 进行自定义配置。以 DistributeTranspilerConfig 为例，修改方式如下所示：

.. code-block:: python

    strategy = DistributedStrategyFactory.create_sync_strategy()

    # 方式一（推荐）：
    config = strategy.get_program_config()
    config.min_block_size = 81920


    # 方式二：调用 set_program_config 修改组网相关配置，支持 DistributeTranspilerConfig 和 dict 两种数据类型
    config = DistributeTranspilerConfig()
    config.min_block_size = 81920
    # config = dict()
    # config['min_block_size'] = 81920
    strategy.set_program_config(config)
