.. _api_guide_cpu_training_best_practice_en:

######################################################
Best practices of distributed training on CPU
######################################################

To improve the training speed of CPU distributed training, we must consider two aspects:

1. Improve the training speed mainly by improving utilization rate of CPU;
2. Improve the communication speed mainly by reducing the amount of data transmitted in the communication;
3. Improve the data IO speed by dataset API;
4. Improve the distributed training speed by changing distributed training strategy.

Improve CPU utilization
=============================

The CPU utilization mainly depends on :code:`ParallelExecutor`, which can make full use of the computing power of multiple CPUs to speed up the calculation.

For detailed API usage, please refer to :ref:`api_fluid_ParallelExecutor` . A simple example:

.. code-block:: python

    # Configure the execution strategy, mainly to set the number of threads
    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = 8

    # Configure the composition strategy, for CPU training, you should use the Reduce mode for training.
    build_strategy = fluid.BuildStrategy()
    if int(os.getenv("CPU_NUM")) > 1:
        build_strategy.reduce_strategy=fluid.BuildStrategy.ReduceStrategy.Reduce

    pe = fluid.ParallelExecutor(
        use_cuda=False,
        loss_name=avg_cost.name,
        main_program=main_program,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)

Among the parameters above:

- :code:`num_threads` : the number of threads used by the model training. It is preferably close to the number of the physical CPU cores of the machine where the training is performed.
- :code:`reduce_strategy` : For CPU training, you should choose fluid.BuildStrategy.ReduceStrategy.Reduce


Configuration of general environment variables:

- :code:`CPU_NUM`: The number of replicas of the model, preferably the same as num_threads


Improve communication speed
==============================

To reduce the amount of communication data and improve communication speed is achieved mainly by using sparse updates, the current support for `sparse update <../layers/sparse_update_en.html>`_ is mainly :ref:`api_fluid_layers_embedding`.

.. code-block:: python

    data = fluid.layers.data(name='ids', shape=[1], dtype='int64')
    fc = fluid.layers.embedding(input=data, size=[dict_size, 16], is_sparse=True)

Among the parameters above:

- :code:`is_sparse`: Use sparse updates to configure embedding. If the dict_size of embedding is large but the number of data are very small each time, it is recommended to use the sparse update method.


Improve data IO speed
==============================

To improve the CPU's distributed training speed, you can first consider using the dataset API as data reader. Dataset is a multi producer and multi consumer data reading method. By default, data reading thread and training thread are coupled. In multi-threaded training, dataset shows a high performance advantage.

Refer to this page for API introduction: https://www.paddlepaddle.org.cn/documentation/docs/en/api/dataset/QueueDataset.html

Combined with the actual model CTR-DNN, you can learn more about how to use dataset: https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleRec/ctr/dnn

Using :code:`train_from_dataset` for network training.

.. code-block:: python

    dataset = fluid.DatasetFactory().create_dataset()
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    exe.train_from_dataset(program=fluid.default_main_program(),dataset=dataset)


Change distributed training strategy
==============================

The core of improving CPU distributed training speed is to choose appropriate distributed training strategy, such as defining communication strategy, compiling strategy, executing strategy and so on. PaddlePaddle released :code:`DistributedStrategy` API in V1.7 version , which can be very flexible and convenient to specify distributed operation strategy.

First, we need to introduce relevant libraries into the code:

.. code-block:: python

    from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
    import paddle.fluid.incubate.fleet.base.role_maker as role_maker
    from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler.distributed_strategy_factory import DistributedStrategyFactory

At present, there are four kinds of training strategies: synchronous training, asynchronous, half asynchronous training and GEO training.


The default configuration of the above policy is introduced by the following code:

.. code-block:: python

    # step1: get distributed strategy
    # Sync
    strategy = DistributedStrategyFactory.create_sync_strategy()
    # Half-Async
    strategy = DistributedStrategyFactory.create_half_async_strategy()
    # Async
    strategy = DistributedStrategyFactory.create_async_strategy()
    # GEO
    strategy = DistributedStrategyFactory.create_geo_strategy(update_frequency=400)

    # step2: define role of node
    role = role_maker.PaddleCloudRoleMaker()
    fleet.init(role)

    # step3: get distributed training program
    optimizer = fluid.optimizer.SGD(learning_rate) # 以 SGD 优化器为例
    optimizer = fleet.distributed_optimizer(optimizer, strategy)
    optimizer.minimize(loss)

    # step4.1: run parameter server node
    if fleet.is_server():
        fleet.init_server()
        fleet.run_server()

    # step4.2: run worker node
    elif fleet.is_worker():
        fleet.init_worker()
        exe.run(fleet.startup_program)
        # Do training
        exe.run(fleet.main_program)
        fleet.stop_worker()

PaddlePaddle supports adjusting the details of the training strategy:

- The build_strategy and exec_strategy which used to create compiled_program can generate from strategy:

.. code-block:: python

    compiled_program = fluid.compiler.CompiledProgram(fleet.main_program).with_data_parallel(
                                                                            loss_name=loss.name,
                                                                            build_strategy=strategy.get_build_strategy(),
                                                                            exec_strategy=strategy.get_execute_strategy())


- Training strategy details can be customized, Paddlepaddle supports customized configuration of distributetranspierconfig, trainerruntimeconfig, serverruntimeconfig, fluid.executionstrategy and fluid.buildstrategy. Take distributetranspillerconfig as an example. The modification method is as follows:

.. code-block:: python

    strategy = DistributedStrategyFactory.create_sync_strategy()

    # Mode 1 (recommended)：
    config = strategy.get_program_config()
    config.min_block_size = 81920


    # Mode 2
    config = DistributeTranspilerConfig()
    config.min_block_size = 81920
    # config = dict()
    # config['min_block_size'] = 81920
    strategy.set_program_config(config)
