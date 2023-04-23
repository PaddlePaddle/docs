.. _api_guide_sync_training:

############
分布式同步训练
############

Fluid 支持数据并行的分布式同步训练，API 使用 :code:`DistributeTranspiler` 将单机网络配置转换成可以多机执行的
:code:`pserver` 端程序和 :code:`trainer` 端程序。用户在不同的节点执行相同的一段代码，根据环境变量或启动参数，
可以执行对应的 :code:`pserver` 或 :code:`trainer` 角色。Fluid 分布式同步训练同时支持 pserver 模式和 NCCL2 模式，
在 API 使用上有差别，需要注意。

pserver 模式分布式训练
===================

API 详细使用方法参考 :ref:`DistributeTranspiler` ，简单实例用法：

.. code-block:: python

    config = fluid.DistributeTranspilerConfig()
    # 配置策略 config
    config.slice_var_up = False
    t = fluid.DistributeTranspiler(config=config)
    t.transpile(trainer_id,
                program=main_program,
                pservers="192.168.0.1:6174,192.168.0.2:6174",
                trainers=1,
                sync_mode=True)

以上参数中：

- :code:`trainer_id` ： trainer 节点的 id，从 0 到 n-1，n 为当前训练任务中 trainer 节点的个数
- :code:`program` ： 被转换的 :code:`program` 默认使用 :code:`fluid.default_main_program()`
- :code:`pservers` ： 当前训练任务中 pserver 节点的 IP 端口列表
- :code:`trainers` ： int 类型，当前训练任务中 trainer 节点的个数。注意：
    * pserver 模式下，trainer 节点个数可以和 pserver 节点个数不一致，比如使用 20 个 pserver 和 50 个 trainer。在实际训练任务中，您可以通过调整 pserver 节点和 trainer 节点个数找到最佳性能
    * NCCL2 模式中，此项参数是字符串，指定 trainer 节点的 IP 端口列表
- :code:`sync_mode` ： 是否是同步训练模式，默认为 True，不传此参数也默认是同步训练模式


其中，支持的 config 包括：

- :code:`slice_var_up` ： 配置是否切分一个参数到多个 pserver 上进行优化，默认开启。此选项适用于模型参数个数少，但需要使用大量节点的场景，有利于提升 pserver 端计算并行度
- :code:`split_method` ： 配置 transpiler 分配参数（或参数的切片）到多个 pserver 的方式，默认为"RoundRobin"，也可以使用"HashName"
- :code:`min_block_size` ： 如果配置了参数切分，指定最小 Tensor 的切分大小，防止 RPC 请求包过小，默认为 8192，一般情况不需要调整此项参数
- :code:`enable_dc_asgd` ： 是否开启 :code:`DC-ASGD` 此选项在异步训练中生效，启用异步训练补偿算法
- :code:`mode` : 可以选择"pserver"或"nccl2"，指定使用 pserver 模式或 NCCL2 模式分布式训练
- :code:`print_log` ： 是否开启 transpiler debug 日志，此项为开发调试使用

通用环境变量配置：

- :code:`FLAGS_rpc_send_thread_num` ：int，指定 RPC 通信发送时线程的个数
- :code:`FLAGS_rpc_get_thread_num` ： int，指定 RPC 通信接受时线程的个数
- :code:`FLAGS_rpc_prefetch_thread_num` ： int，分布式 lookup table 执行 RPC 通信时，prefetch 线程的个数
- :code:`FLAGS_rpc_deadline` ： int，RPC 通信最长等待时间，单位为毫秒，默认 180000


NCCL2 模式分布式训练
=================

基于 NCCL2 (Collective Communication) 的多机同步训练模式，仅支持在 GPU 集群下进行。
此部分详细 API 说明可以参考 :ref:`DistributeTranspiler` 。

注意：NCCL2 模式下，集群不需要启动 pserver，只需要启动多个 trainer 节点即可。

使用以下代码，将当前 :code:`Program` 转化成适用于 NCCL2 分布式计算的 Fluid :code:`Program` ：

.. code-block:: python

    config = fluid.DistributeTranspilerConfig()
    config.mode = "nccl2"
    t = fluid.DistributeTranspiler(config=config)
    t.transpile(trainer_id,
                program=main_program,
                startup_program=startup_program,
                trainers="192.168.0.1:6174,192.168.0.2:6174",
                current_endpoint="192.168.0.1:6174")

其中：

- :code:`trainer_id` : trainer 节点的 id，从 0 到 n-1，n 为当前训练任务中 trainer 节点的个数
- :code:`program` 和 :code:`startup_program` : 分别为 Fluid 模型的主配置 program 和初始化 startup_program
- :code:`trainers` : 字符串类型，指定当前任务所有 trainer 的 IP 和端口号，仅用于 NCCL2 初始化（pserver 模式中，此参数为 int，指定 trainer 节点的个数）
- :code:`current_endpoint` : 当前任务的当前节点的 IP 和端口号
