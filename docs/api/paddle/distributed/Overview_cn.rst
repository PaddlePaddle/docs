.. _cn_overview_distributed:

paddle.distributed
============================

paddle.distributed 目录包含的 API 支撑飞桨框架大规模分布式训练能力。具体如下：

-  :ref:`Fleet 分布式高层 API <01>`
-  :ref:`环境配置和训练启动管理 <02>`
-  :ref:`数据加载 <03>`
-  :ref:`集合通信算法 API <04>`
-  :ref:`Stream 集合通信高级 API <05>`
-  :ref:`RPC API <06>`
-  :ref:`自动并行 API <07>`
-  :ref:`Sharding API <08>`

.. _01:

Fleet 分布式高层 API
::::::::::::::::::::::::::

``paddle.distributed.fleet`` 是分布式训练的统一入口 API，用于配置分布式训练。

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 20, 50

    " :ref:`UserDefinedRoleMaker <cn_api_paddle_distributed_fleet_UserDefinedRoleMaker>` ", "设置和获取用户自定义的集群信息，支持集合通信（Collective）及参数服务器（ParameterServer）两种训练架构的初始化"
    " :ref:`PaddleCloudRoleMaker <cn_api_paddle_distributed_fleet_PaddleCloudRoleMaker>` ", "设置和获取 paddlecloud 集群信息（百度内部集群使用），支持集合通信（Collective）及参数服务器（ParameterServer）两种训练架构的初始化"
    " :ref:`DistributedStrategy <cn_api_paddle_distributed_fleet_DistributedStrategy>` ", "配置分布式通信、计算和内存优化等策略"
    " :ref:`fleet.init <cn_api_paddle_distributed_fleet_Fleet>` ", "进行分布式训练配置并初始化 "
    " :ref:`fleet.init_worker <cn_api_paddle_distributed_fleet_Fleet>` ", "集合通信架构下，worker 节点初始化 "
    " :ref:`fleet.stop_worker <cn_api_paddle_distributed_fleet_Fleet>` ", "集合通信架构下，停止正在运行的 worker 节点"
    " :ref:`fleet.barrier_worker <cn_api_paddle_distributed_fleet_Fleet>` ", "集合通信架构下，强制要求所有的 worker 在此处相互等待一次，保持同步"
    " :ref:`fleet.init_server <cn_api_paddle_distributed_fleet_Fleet>` ", "参数服务器架构下，server 节点的初始化  "
    " :ref:`fleet.run_server <cn_api_paddle_distributed_fleet_Fleet>` ", "参数服务器架构下的进程启动"
    " :ref:`fleet.save_inference_model <cn_api_paddle_distributed_fleet_Fleet>` ", "保存用于预测的模型"
    " :ref:`fleet.save_persistables <cn_api_paddle_distributed_fleet_Fleet>` ", "保存全量模型参数"
    " :ref:`fleet.distributed_optimizer <cn_api_paddle_distributed_fleet_Fleet>` ", "基于分布式并行策略进行模型拆分和优化计算"
    " :ref:`UtilBase <cn_api_paddle_distributed_fleet_UtilBase>` ", "分布式训练工具的基类，用户集合通信、文件系统操作"
    " :ref:`utils.HDFSClient <cn_api_paddle_distributed_fleet_utils_HDFSClient>` ", "Hadoop 文件系统查看和管理"
    " :ref:`utils.LocalFS <cn_api_paddle_distributed_fleet_utils_LocalFS>` ", "本地文件系统查看和管理"
    " :ref:`utils.recompute <cn_api_paddle_distributed_fleet_utils_recompute>` ", "重新计算中间激活函数值来节省显存"

.. _02:

环境配置和训练启动管理
::::::::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 20, 50

    " :ref:`is_initialized <cn_api_paddle_distributed_is_initialized>` ", "检查分布式环境是否已经被初始化"
    " :ref:`is_available <cn_api_paddle_distributed_is_available>` ", "检查分布式环境是否可用"
    " :ref:`init_parallel_env <cn_api_paddle_distributed_init_parallel_env>` ", "初始化并行训练环境，支持动态图模式"
    " :ref:`launch <cn_api_paddle_distributed_launch>` ", "启动分布式训练进程，支持集合通信及参数服务器架构"
    " :ref:`spawn <cn_api_paddle_distributed_spawn>` ", "启动分布式训练进程，仅支持集合通信架构"
    " :ref:`get_rank <cn_api_paddle_distributed_get_rank>` ", "获取当前进程的 rank 值"
    " :ref:`get_world_size <cn_api_paddle_distributed_get_world_size>` ", "获取当前进程数"
    " :ref:`new_group <cn_api_paddle_distributed_new_group>` ", "创建分布式通信组"
    " :ref:`get_group <cn_api_paddle_distributed_get_group>` ", "通过通信组 id 获取通信组实例"
    " :ref:`destroy_process_group <cn_api_paddle_distributed_destroy_process_group>` ", "销毁分布式通信组"
    " :ref:`get_backend <cn_api_paddle_distributed_get_backend>` ", "获取指定分布式通信组后端的名称"
    " :ref:`gloo_init_parallel_env <cn_api_paddle_distributed_gloo_init_parallel_env>` ", "初始化 ``GLOO`` 上下文用于 CPU 间的通信"
    " :ref:`gloo_release <cn_api_paddle_distributed_gloo_release>` ", "释放当前并行环境的 gloo 上下文"
    " :ref:`ParallelEnv <cn_api_paddle_distributed_ParallelEnv>` ", "这个类用于获取动态图模型并行执行所需的环境变量值"

.. _03:

数据加载
::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 20, 50


    " :ref:`InMemoryDataset <cn_api_paddle_distributed_InMemoryDataset>` ", "数据加载到内存中，在训练前随机整理数据"
    " :ref:`QueueDataset <cn_api_paddle_distributed_QueueDataset>` ", "流式数据加载"

.. _04:

集合通信 API
::::::::::::::::::::::

在集群上，对多设备的进程组的参数数据 tensor 或 object 进行计算处理，包括规约、聚合、广播、分发等。

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 20, 50

    " :ref:`ReduceOp <cn_api_paddle_distributed_ReduceOp>` ", "规约操作的类型"
    " :ref:`reduce <cn_api_paddle_distributed_reduce>` ", "规约进程组内的 tensor，随后将结果发送到指定进程"
    " :ref:`all_reduce <cn_api_paddle_distributed_all_reduce>` ", "规约进程组内的 tensor，随后将结果发送到每个进程"
    " :ref:`all_gather <cn_api_paddle_distributed_all_gather>` ", "聚合进程组内的 tensor，随后将结果发送到每个进程"
    " :ref:`all_gather_object <cn_api_paddle_distributed_all_gather_object>` ", "聚合进程组内的 object，随后将结果发送到每个进程"
    " :ref:`alltoall <cn_api_paddle_distributed_alltoall>` ", "将一组 tensor 分发到每个进程并进行聚合"
    " :ref:`alltoall_single <cn_api_paddle_distributed_alltoall_single>` ", "将一个 tensor 分发到每个进程并进行聚合"
    " :ref:`broadcast <cn_api_paddle_distributed_broadcast>` ", "将一个 tensor 发送到每个进程"
    " :ref:`broadcast_object_list <cn_api_paddle_distributed_broadcast_object_list>` ", "将一组 object 发送到每个进程"
    " :ref:`scatter <cn_api_paddle_distributed_scatter>` ", "将一组 tensor 分发到每个进程"
    " :ref:`scatter_object_list <cn_api_paddle_distributed_scatter_object_list>` ", "将一组 object 分发到每个进程"
    " :ref:`reduce_scatter <cn_api_paddle_distributed_reduce_scatter>` ", "规约一组 tensor，随后将规约结果分发到每个进程"
    " :ref:`isend <cn_api_paddle_distributed_isend>` ", "异步发送一个 tensor 到指定进程"
    " :ref:`irecv <cn_api_paddle_distributed_irecv>` ", "异步接收一个来自指定进程的 tensor"
    " :ref:`send <cn_api_paddle_distributed_send>` ", "发送一个 tensor 到指定进程"
    " :ref:`recv <cn_api_paddle_distributed_recv>` ", "接收一个来自指定进程的 tensor"
    " :ref:`barrier <cn_api_paddle_distributed_barrier>` ", "同步路障，阻塞操作以实现组内进程同步"
    " :ref:`gloo_barrier <cn_api_paddle_distributed_gloo_barrier>` ", "使用初始化的 gloo 上下文直接调用基于 gloo 封装的 barrier 函数"
    " :ref:`wait <cn_api_paddle_distributed_wait>` ", "同步通信组，在指定的通信组中同步特定的 tensor 对象"

.. _05:

Stream 集合通信高级 API
::::::::::::::::::::::

``paddle.distributed.stream`` 在集合通信 API 的基础上，提供更统一的语义和对计算流的更精细的控制能力，有助于在特定场景下提高性能。

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 25, 50


    " :ref:`stream.reduce <cn_api_paddle_distributed_stream_reduce>` ", "规约进程组内的 tensor，随后将结果发送到指定进程"
    " :ref:`stream.all_reduce <cn_api_paddle_distributed_stream_all_reduce>` ", "规约进程组内的 tensor，随后将结果发送到每个进程"
    " :ref:`stream.all_gather <cn_api_paddle_distributed_stream_all_gather>` ", "聚合进程组内的 tensor，随后将结果发送到每个进程"
    " :ref:`stream.alltoall <cn_api_paddle_distributed_stream_alltoall>` ", "分发一组 tensor 到每个进程并进行聚合"
    " :ref:`stream.alltoall_single <cn_api_paddle_distributed_stream_alltoall_single>` ", "分发一个 tensor 到每个进程并进行聚合"
    " :ref:`stream.broadcast <cn_api_paddle_distributed_stream_broadcast>` ", "发送一个 tensor 到每个进程"
    " :ref:`stream.scatter <cn_api_paddle_distributed_stream_scatter>` ", "分发一个 tensor 到每个进程"
    " :ref:`stream.reduce_scatter <cn_api_paddle_distributed_stream_reduce_scatter>` ", "规约一组 tensor，随后将规约结果分发到每个进程"
    " :ref:`stream.send <cn_api_paddle_distributed_stream_send>` ", "发送一个 tensor 到指定进程"
    " :ref:`stream.recv <cn_api_paddle_distributed_stream_recv>` ", "接收一个来自指定进程的 tensor"

.. _06:

RPC API
::::::::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 20, 50


    " :ref:`rpc.init_rpc <cn_api_paddle_distributed_rpc_init_rpc>` ", "初始化 RPC "
    " :ref:`rpc.rpc_sync <cn_api_paddle_distributed_rpc_rpc_sync>` ", "发起一个阻塞的 RPC 调用"
    " :ref:`rpc.rpc_async <cn_api_paddle_distributed_rpc_rpc_async>` ", "发起一个非阻塞的 RPC 调用"
    " :ref:`rpc.shutdown <cn_api_paddle_distributed_rpc_shutdown>` ", "关闭 RPC "
    " :ref:`rpc.get_worker_info <cn_api_paddle_distributed_rpc_get_worker_info>` ", "获取 worker 信息"
    " :ref:`rpc.get_all_worker_infos <cn_api_paddle_distributed_rpc_get_all_worker_infos>` ", "获取所有 worker 的信息"
    " :ref:`rpc.get_current_worker_info <cn_api_paddle_distributed_rpc_get_current_worker_info>` ", "获取当前 worker 的信息"

.. _07:

自动并行 API
::::::::::::::::::::::::::

自动并行降低分布式训练的使用门槛，使用自动并行 API 对组网进行少量改动即可进行分布式训练。

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 20, 50

    " :ref:`shard_tensor <cn_api_paddle_distributed_shard_tensor>` ", "创建带有分布式切分信息的分布式 Tensor"
    " :ref:`dtensor_from_fn <cn_api_paddle_distributed_dtensor_from_fn>` ", "通过一个 paddle API 结合分布式属性 placements 创建一个带分布式属性的 Tensor"
    " :ref:`shard_layer <cn_api_paddle_distributed_shard_layer>` ", "按照指定方式将 Layer 中的参数转换为分布式 Tensor"
    " :ref:`reshard <cn_api_paddle_distributed_reshard>`", "对一个带有分布式信息的 Tensor 重新进行分布/切片"
    " :ref:`to_static <cn_api_paddle_distributed_to_static>`", "将带有分布式切分信息的动态图模型转换为静态图分布式模型"
    " :ref:`Strategy <cn_api_paddle_distributed_Strategy>`", "配置静态图分布式训练时所使用的并行策略和优化策略"
    " :ref:`DistAttr <cn_api_paddle_distributed_DistAttr>` ", "指定 Tensor 在 ProcessMesh 上的分布或切片方式"
    " :ref:`shard_optimizer <cn_api_paddle_distributed_shard_optimizer>` ", "将单卡视角的优化器转变为分布式视角"
    " :ref:`split <cn_api_paddle_distributed_split>` ", "切分指定操作的参数到多个设备，并且并行计算得到结果"
    " :ref:`set_mesh <cn_api_paddle_distributed_set_mesh>` ", "设置全局 ProcessMesh"
    " :ref:`get_mesh <cn_api_paddle_distributed_get_mesh>` ", "获取全局 ProcessMesh"


此外，自动并行提供更高层次的 API 来帮助用户通过非入侵组网的方式实现自动并行的分布式训练。

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 20, 50

    " :ref:`parallelize <cn_api_paddle_distributed_parallelize>` ", "对模型和优化器进行并行化处理"
    " :ref:`ColWiseParallel <cn_api_paddle_distributed_ColWiseParallel>` ", "按列切分标识 Layer"
    " :ref:`RowWiseParallel <cn_api_paddle_distributed_RowWiseParallel>` ", "按行切分标识 Layer"
    " :ref:`SequenceParallelBegin <cn_api_paddle_distributed_SequenceParallelBegin>` ", "标识 Layer 为序列并行的开始"
    " :ref:`SequenceParallelEnd <cn_api_paddle_distributed_SequenceParallelEnd>` ", "标识 Layer 序列并行的结束"
    " :ref:`SequenceParallelEnable <cn_api_paddle_distributed_SequenceParallelEnable>` ", "对标识 Layer 进行序列并行"
    " :ref:`SequenceParallelDisable <cn_api_paddle_distributed_SequenceParallelDisable>` ", "对标识 Layer 不进行序列并行"
    " :ref:`SplitPoint <cn_api_paddle_distributed_SplitPoint>` ", "标识 Layer 为流水线并行的切分点"
    " :ref:`PrepareLayerInput <cn_api_paddle_distributed_PrepareLayerInput>` ", "对标识 Layer 的输入进行处理"
    " :ref:`PrepareLayerOutput <cn_api_paddle_distributed_PrepareLayerOutput>` ", "对标识 Layer 的输出进行处理"



.. _08:

Sharding API
::::::::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 20, 50

    " :ref:`sharding.group_sharded_parallel <cn_api_paddle_distributed_sharding_group_sharded_parallel>`", "对模型、优化器和 GradScaler 做 group sharded 配置"
    " :ref:`sharding.save_group_sharded_model <cn_api_paddle_distributed_sharding_save_group_sharded_model>`", "对 group_sharded_parallel 配置后的模型和优化器状态进行保存"
