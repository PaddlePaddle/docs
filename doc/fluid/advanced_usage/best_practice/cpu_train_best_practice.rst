.. _api_guide_cpu_training_best_practice:

####################
分布式CPU训练最佳实践
####################

提高CPU分布式训练的训练速度，主要要从两个方面来考虑：
1）提高训练速度，主要是提高CPU的使用率；2）提高通信速度，主要是减少通信传输的数据量。

提高CPU的使用率
=============

提高CPU使用率主要依赖 :code:`ParallelExecutor`，可以充分利用多个CPU的计算能力来加速计算。

API详细使用方法参考 :ref:`cn_api_fluid_ParallelExecutor` ，简单实例用法：

.. code-block:: python

    # 配置执行策略，主要是设置线程数
    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = 8

    # 配置构图策略，对于CPU训练而言，应该使用Reduce模式进行训练
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

- :code:`num_threads` ： 模型训练使用的线程数，最好和训练所在机器的物理CPU核数接近
- :code:`reduce_strategy` ： 对于CPU训练而言，应该选择 fluid.BuildStrategy.ReduceStrategy.Reduce


通用环境变量配置：

- :code:`CPU_NUM` ：模型副本replica的个数，最好和num_threads一致


提高通信速度
==========

要减少通信数据量，提高通信速度，主要是使用稀疏更新 ，目前支持 `稀疏更新 <../layers/sparse_update.html>`_  的主要是  :ref:`cn_api_fluid_layers_embedding` 。

.. code-block:: python

    data = fluid.layers.data(name='ids', shape=[1], dtype='int64')
    fc = fluid.layers.embedding(input=data, size=[dict_size, 16], is_sparse=True)

以上参数中：

- :code:`is_sparse` ： 配置embedding使用稀疏更新，如果embedding的dict_size很大，而每次数据data很少，建议使用sparse更新方式。
