.. _api_guide_sync_training:

############
分布式同步训练
############

Fluid支持数据并行的分布式同步训练，API使用 :code:`DistributedTranspiler` 将单机网络配置转换成可以多机执行的
:code:`pserver` 端程序和 :code:`trainer` 端程序，用户在不同的节点执行相同的一段代码，根据环境变量或启动参数，
可以执行对应的 :code:`pserver` 或 :code:`trainer` 角色。Fluid分布式同步训练同时支持pserver模式和NCCL2模式，
在API使用上有差别，需要注意。

pserver模式分布式训练
===================

使用DistributedTranspiler
++++++++++++++++++++++++

简单用法
.. code-block:: python

    t = fluid.DistributedTranspiler(config=config)
    t.transpile(trainer_id, 
                program=main_program,
                pservers="192.168.0.1:6174,192.168.0.2:6174",
                trainers=1,
                sync_mode=True)

以上参数中：
- :code:`trainer_id` : trainer节点的id，从0到n-1，n为当前训练任务中trainer节点的个数
- :code:`program` : 被转换的 :code:`program` 默认使用 :code:`fluid.default_main_program()`
- :code:`pservers` : 当前训练任务中pserver节点的IP端口列表
- :code:`trainers` : 当前训练任务中trainer节点的个数（注意NCCL2模式中，此项参数是字符串，指定trainer节点的IP端口列表），
  注意，在pserver模式下，trainer节点个数可以和pserver节点个数不一致，比如使用20个pserver和50个trainer。在实际训练任务
  中，您可以通过调整pserver节点和trainer节点个数找到最佳性能。
- :code:`sync_mode` : 是否是同步训练模式，默认为True，不传此参数也默认是同步训练模式

.. code-block:: python

    config = fluid.DistributedTranspilerConfig()
    # 配置策略config
    config.slice_var_up = False
    t = fluid.DistributedTranspiler(config=config)

其中，支持的config包括：

- :code:`slice_var_up` : 配置是否切分一个参数到多个pserver上进行优化，默认开启。此选项适用于模型参数个数少，
  但需要使用大量节点的场景，有利于提升pserver端计算并行度
- :code:`split_method` : 配置transpiler分配参数（或参数的切片）到多个pserver的方式，
  默认为"RoundRobin"，也可以使用"HashName"
- :code:`min_block_size` : 如果配置了参数切分，指定最小Tensor的切分大小，防止RPC请求包过小，默认为8192，一般情况
  不需要调整此项参数
- :code:`enable_dc_asgd` : 是否开启 :code:`DC-ASGD` 此选项在异步训练中生效，启用异步训练补偿算法
- :code:`mode` : 可以选择"pserver"或"nccl2"，指定使用pserver模式或NCCL2模式分布式训练
- :code:`print_log` : 是否开启transpiler debug日志，此项为开发调试使用

NCCL2模式分布式训练
=================

TBD