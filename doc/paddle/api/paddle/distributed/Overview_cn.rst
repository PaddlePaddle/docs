.. _cn_overview_distributed:

paddle.distributed
============================

paddle.distributed目录包含的API支撑飞桨框架大规模分布式训练能力。具体如下：

-  :ref:`Fleet分布式高层API <01>`
-  :ref:`环境配置和训练启动管理 <02>`
-  :ref:`数据加载 <03>`
-  :ref:`集合通信算法API <04>`

.. _01:

Fleet分布式高层API
::::::::::::::::::::::::::

paddle.distributed.fleet是分布式训练的统一入口API，用于配置分布式训练。

.. csv-table::
    :header: "API名称", "API功能"
    :widths: 20, 50

    " :ref:`UserDefinedRoleMaker <cn_api_distributed_fleet_UserDefinedRoleMaker>` ", "设置和获取用户自定义的集群信息，支持集合通信"
    " :ref:`PaddleCloudRoleMaker <cn_api_distributed_fleet_PaddleCloudRoleMaker>` ", "设置和获取paddlecloud集群信息（百度内部集群使用），支持集合通信（Collective）及参数服务器（ParameterServer）两种训练架构的初始化"
    " :ref:`DistributedStrategy <cn_api_distributed_fleet_DistributedStrategy>` ", "配置分布式通信、计算和内存优化等策略"
    " :ref:`fleet.init <cn_api_distributed_fleet_Fleet>` ", "进行分布式训练配置并初始化 "
    " :ref:`fleet.init_worker <cn_api_distributed_fleet_Fleet>` ", "集合通信架构下，worker节点初始化 "
    " :ref:`fleet.stop_worker <cn_api_distributed_fleet_Fleet>` ", "集合通信架构下，停止正在运行的worker节点"
    " :ref:`fleet.barrier_worker <cn_api_distributed_fleet_Fleet>` ", "集合通信架构下，强制要求所有的worker在此处相互等待一次，保持同步"
    " :ref:`fleet.init_server <cn_api_distributed_fleet_Fleet>` ", "参数服务器架构下，server节点的初始化  "
    " :ref:`fleet.run_server <cn_api_distributed_fleet_Fleet>` ", "参数服务器架构下的进程启动"
    " :ref:`fleet.save_inference_model <cn_api_distributed_fleet_Fleet>` ", "保存用于预测的模型"
    " :ref:`fleet.save_persistables <cn_api_distributed_fleet_Fleet>` ", "保存全量模型参数"
    " :ref:`fleet.distributed_optimizer <cn_api_distributed_fleet_Fleet>` ", "基于分布式并行策略进行模型拆分和优化计算"
    " :ref:`UtilBase <cn_api_distributed_fleet_UtilBase>` ", "分布式训练工具的基类，用户集合通信、文件系统操作"
    " :ref:`utils.HDFSClient <cn_api_distributed_fleet_utils_fs_HDFSClient>` ", "Hadoop文件系统查看和管理"
    " :ref:`utils.LocalFS <cn_api_distributed_fleet_utils_fs_LocalFS>` ", "本地文件系统查看和管理"

.. _02:

环境配置和训练启动管理
::::::::::::::::::::::::::

.. csv-table::
    :header: "API名称", "API功能"
    :widths: 20, 50
    

    " :ref:`init_parallel_env <cn_api_distributed_init_parallel_env>` ", "初始化并行训练环境，支持动态图模式"
    " :ref:`launch <>` ", "启动分布式训练进程，支持集合通信及参数服务器架构"
    " :ref:`spawn <cn_api_distributed_spawn>` ", "启动分布式训练进程，仅支持集合通信架构"
    " :ref:`get_rank <cn_api_distributed_get_rank>` ", "获取当前进程的rank值"
    " :ref:`get_world_size <cn_api_distributed_get_world_size>` ", "获取当前进程数"

.. _03:

数据加载
::::::::::::::

.. csv-table::
    :header: "API名称", "API功能"
    :widths: 20, 50
    

    " :ref:`InMemoryDataset <cn_api_distributed_InMemoryDataset>` ", "数据加载到内存中，在训练前随机整理数据"
    " :ref:`QueueDataset <cn_api_distributed_QueueDataset>` ", "流式数据加载"

.. _04:

集合通信算法API
::::::::::::::::::::::

在集群上，对多设备的进程组的参数数据tensor进行计算处理。

.. csv-table::
    :header: "API名称", "API功能"
    :widths: 20, 50
    

    " :ref:`reduce <cn_api_distributed_reduce>` ", "规约，规约进程组内的tensor，返回结果至指定进程"
    " :ref:`ReduceOP <cn_api_distributed_ReduceOp>` ", "规约，指定逐元素规约操作"
    " :ref:`all_reduce <cn_api_distributed_all_reduce>` ", "组规约，规约进程组内的tensor，结果广播至每个进程"
    " :ref:`all_gather <cn_api_distributed_all_gather>` ", "组聚合，聚合进程组内的tensor，结果广播至每个进程"
    " :ref:`broadcast <cn_api_distributed_broadcast>` ", "广播一个tensor到每个进程"
    " :ref:`scatter <cn_api_distributed_scatter>` ", "分发tensor到每个进程"
    " :ref:`split <cn_api_distributed_split>` ", "切分参数到多个设备"
    " :ref:`barrier <cn_api_distributed_barrier>` ", "同步路障，进行阻塞操作，实现组内所有进程的同步"
