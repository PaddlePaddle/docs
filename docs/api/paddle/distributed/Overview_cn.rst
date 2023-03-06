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

.. _01:

Fleet 分布式高层 API
::::::::::::::::::::::::::

``paddle.distributed.fleet`` 是分布式训练的统一入口 API，用于配置分布式训练。

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 20, 50

    " :ref:`UserDefinedRoleMaker <cn_api_distributed_fleet_UserDefinedRoleMaker>` ", "设置和获取用户自定义的集群信息，支持集合通信（Collective）及参数服务器（ParameterServer）两种训练架构的初始化"
    " :ref:`PaddleCloudRoleMaker <cn_api_distributed_fleet_PaddleCloudRoleMaker>` ", "设置和获取 paddlecloud 集群信息（百度内部集群使用），支持集合通信（Collective）及参数服务器（ParameterServer）两种训练架构的初始化"
    " :ref:`DistributedStrategy <cn_api_distributed_fleet_DistributedStrategy>` ", "配置分布式通信、计算和内存优化等策略"
    " :ref:`fleet.init <cn_api_distributed_fleet_Fleet>` ", "进行分布式训练配置并初始化 "
    " :ref:`fleet.init_worker <cn_api_distributed_fleet_Fleet>` ", "集合通信架构下，worker 节点初始化 "
    " :ref:`fleet.stop_worker <cn_api_distributed_fleet_Fleet>` ", "集合通信架构下，停止正在运行的 worker 节点"
    " :ref:`fleet.barrier_worker <cn_api_distributed_fleet_Fleet>` ", "集合通信架构下，强制要求所有的 worker 在此处相互等待一次，保持同步"
    " :ref:`fleet.init_server <cn_api_distributed_fleet_Fleet>` ", "参数服务器架构下，server 节点的初始化  "
    " :ref:`fleet.run_server <cn_api_distributed_fleet_Fleet>` ", "参数服务器架构下的进程启动"
    " :ref:`fleet.save_inference_model <cn_api_distributed_fleet_Fleet>` ", "保存用于预测的模型"
    " :ref:`fleet.save_persistables <cn_api_distributed_fleet_Fleet>` ", "保存全量模型参数"
    " :ref:`fleet.distributed_optimizer <cn_api_distributed_fleet_Fleet>` ", "基于分布式并行策略进行模型拆分和优化计算"
    " :ref:`UtilBase <cn_api_distributed_fleet_UtilBase>` ", "分布式训练工具的基类，用户集合通信、文件系统操作"
    " :ref:`utils.HDFSClient <cn_api_distributed_fleet_utils_fs_HDFSClient>` ", "Hadoop 文件系统查看和管理"
    " :ref:`utils.LocalFS <cn_api_distributed_fleet_utils_fs_LocalFS>` ", "本地文件系统查看和管理"

.. _02:

环境配置和训练启动管理
::::::::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 20, 50

    " :ref:`is_available <cn_api_distributed_is_available>` ", "检查分布式环境是否可用"
    " :ref:`init_parallel_env <cn_api_distributed_init_parallel_env>` ", "初始化并行训练环境，支持动态图模式"
    " :ref:`launch <cn_api_distributed_launch>` ", "启动分布式训练进程，支持集合通信及参数服务器架构"
    " :ref:`spawn <cn_api_distributed_spawn>` ", "启动分布式训练进程，仅支持集合通信架构"
    " :ref:`get_rank <cn_api_distributed_get_rank>` ", "获取当前进程的 rank 值"
    " :ref:`get_world_size <cn_api_distributed_get_world_size>` ", "获取当前进程数"
    " :ref:`new_group <cn_api_distributed_new_group>` ", "创建分布式通信组"
    " :ref:`destroy_process_group <cn_api_distributed_destroy_process_group>` ", "销毁分布式通信组"
    " :ref:`get_backend <cn_api_distributed_get_backend>` ", "获取指定分布式通信组后端的名称"

.. _03:

数据加载
::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 20, 50


    " :ref:`InMemoryDataset <cn_api_distributed_InMemoryDataset>` ", "数据加载到内存中，在训练前随机整理数据"
    " :ref:`QueueDataset <cn_api_distributed_QueueDataset>` ", "流式数据加载"

.. _04:

集合通信 API
::::::::::::::::::::::

在集群上，对多设备的进程组的参数数据 tensor 或 object 进行计算处理，包括规约、聚合、广播、分发等。

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 20, 50

    " :ref:`ReduceOp <cn_api_distributed_ReduceOp>` ", "规约操作的类型"
    " :ref:`reduce <cn_api_distributed_reduce>` ", "规约进程组内的 tensor，随后将结果发送到指定进程"
    " :ref:`all_reduce <cn_api_distributed_all_reduce>` ", "规约进程组内的 tensor，随后将结果发送到每个进程"
    " :ref:`all_gather <cn_api_distributed_all_gather>` ", "聚合进程组内的 tensor，随后将结果发送到每个进程"
    " :ref:`all_gather_object <cn_api_distributed_all_gather_object>` ", "聚合进程组内的 object，随后将结果发送到每个进程"
    " :ref:`alltoall <cn_api_distributed_alltoall>` ", "将一组 tensor 分发到每个进程并进行聚合"
    " :ref:`alltoall_single <cn_api_distributed_alltoall_single>` ", "将一个 tensor 分发到每个进程并进行聚合"
    " :ref:`broadcast <cn_api_distributed_broadcast>` ", "将一个 tensor 发送到每个进程"
    " :ref:`broadcast_object_list <cn_api_distributed_broadcast_object_list>` ", "将一组 object 发送到每个进程"
    " :ref:`scatter <cn_api_distributed_scatter>` ", "将一组 tensor 分发到每个进程"
    " :ref:`scatter_object_list <cn_api_distributed_scatter_object_list>` ", "将一组 object 分发到每个进程"
    " :ref:`reduce_scatter <cn_api_distributed_reduce_scatter>` ", "规约一组 tensor，随后将规约结果分发到每个进程"
    " :ref:`isend <cn_api_distributed_isend>` ", "异步发送一个 tensor 到指定进程"
    " :ref:`irecv <cn_api_distributed_irecv>` ", "异步接收一个来自指定进程的 tensor"
    " :ref:`send <cn_api_distributed_send>` ", "发送一个 tensor 到指定进程"
    " :ref:`recv <cn_api_distributed_recv>` ", "接收一个来自指定进程的 tensor"
    " :ref:`barrier <cn_api_distributed_barrier>` ", "同步路障，阻塞操作以实现组内进程同步"

.. _05:

Stream 集合通信高级 API
::::::::::::::::::::::

``paddle.distributed.stream`` 在集合通信 API 的基础上，提供更统一的语义和对计算流的更精细的控制能力，有助于在特定场景下提高性能。

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 25, 50


    " :ref:`stream.reduce <cn_api_distributed_stream_reduce>` ", "规约进程组内的 tensor，随后将结果发送到指定进程"
    " :ref:`stream.all_reduce <cn_api_distributed_stream_all_reduce>` ", "规约进程组内的 tensor，随后将结果发送到每个进程"
    " :ref:`stream.all_gather <cn_api_distributed_stream_all_gather>` ", "聚合进程组内的 tensor，随后将结果发送到每个进程"
    " :ref:`stream.alltoall <cn_api_distributed_stream_alltoall>` ", "分发一组 tensor 到每个进程并进行聚合"
    " :ref:`stream.alltoall_single <cn_api_distributed_stream_alltoall_single>` ", "分发一个 tensor 到每个进程并进行聚合"
    " :ref:`stream.broadcast <cn_api_distributed_stream_broadcast>` ", "发送一个 tensor 到每个进程"
    " :ref:`stream.scatter <cn_api_distributed_stream_scatter>` ", "分发一个 tensor 到每个进程"
    " :ref:`stream.reduce_scatter <cn_api_distributed_stream_reduce_scatter>` ", "规约一组 tensor，随后将规约结果分发到每个进程"
    " :ref:`stream.send <cn_api_distributed_stream_send>` ", "发送一个 tensor 到指定进程"
    " :ref:`stream.recv <cn_api_distributed_stream_recv>` ", "接收一个来自指定进程的 tensor"

.. _06:

RPC API
::::::::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 20, 50


    " :ref:`rpc.init_rpc <cn_api_distributed_rpc_init_rpc>` ", "初始化 RPC "
    " :ref:`rpc.rpc_sync <cn_api_distributed_rpc_rpc_sync>` ", "发起一个阻塞的 RPC 调用"
    " :ref:`rpc.rpc_async <cn_api_distributed_rpc_rpc_async>` ", "发起一个非阻塞的 RPC 调用"
    " :ref:`rpc.shutdown <cn_api_distributed_rpc_shutdown>` ", "关闭 RPC "
    " :ref:`rpc.get_worker_info <cn_api_distributed_rpc_get_worker_info>` ", "获取 worker 信息"
    " :ref:`rpc.get_all_worker_infos <cn_api_distributed_rpc_get_all_worker_infos>` ", "获取所有 worker 的信息"
    " :ref:`rpc.get_current_worker_info <cn_api_distributed_rpc_get_current_worker_info>` ", "获取当前 worker 的信息"
