飞桨分布式整体介绍
==================

近十年来，深度学习技术不断刷新视觉、自然语言处理、语音、搜索和推荐等领域任务的记录。这其中的原因，用一个关键词描述就是“大规模”。大规模的数据使得模型有足够的知识可以记忆，大规模参数量的模型使得模型本身有能力记忆更多的数据，大规模高性能的算力（以GPU为典型代表）使得模型的训练速度有百倍甚至千倍的提升。大规模的数据、模型和算力作为深度学习技术的基石，在推动深度学习技术发展的同时，也给深度学习训练带来了新的挑战：大规模数据和大规模模型的发展使得深度学习模型的能力不断增强，如何更加合理地利用大规模集群算力高效地训练，这是分布式训练需要解决的问题。

飞桨分布式从产业实践出发，提供参数服务器(Parameter Server)和集合通信(Collective)两种主流分布式训练构架，包含丰富的并行能力，提供简单易用地分布式训练接口和丰富的底层通信原语，赋能用户业务发展。

参数服务器架构
~~~~~~~~~~~~~~~~~~~~~~~~

参数服务器是一种编程范式，方便用户分布式编程。参数服务器架构的重点是对模型参数的分布式存储和协同支持。参数服务器架构如下图所示，集群中的节点分为两种角色：计算节点（Worker）和参数服务器节点（Server）。

- Worker 负责从参数服务节点拉取参数，根据分配给自己的训练数据计算得到参数梯度，并将梯度推送给对应的 Server。
- Server 负责存储参数，并采用分布式存储的方式各自存储全局参数的一部分，同时响应 Worker 的查询请求并更新参数。

.. image:: https://github.com/PaddlePaddle/FleetX/blob/develop/docs/source/paddle_fleet_rst/collective/img/ps_arch.png?raw=true
  :alt: Parameter-Server Architecture
  :align: center

具体地讲，参数服务器架构下，模型参数分配到所有的 Server ，即每个 Server 上只保存部分模型参数。在高可靠性要求场景下，也可以将每个参数备份在多个 Server 。每个 Worker 上的计算算子都是相同的（即数据并行），完整的数据集被切分到每个 Worker ，每个 Worker 使用本地分配的数据进行计算：在每次迭代中，Worker 从 Server 拉取参数用于训练本地模型，计算完成后得到对应参数的梯度，并把梯度上传给 Server 以更新参数。Server 获取 Worker 传输的梯度后，汇总并更新参数。

集合通信架构
~~~~~~~~~~~~~~~

与参数服务器架构具有两种角色不同，集合通信架构中所有的训练节点是对等的，可以说都是Worker。节点间通过Collective集合通信功能通信，因此也称为Collective训练，如下图所示。一种典型的集合通信功能实现是基于\ `NVIDIA NCCL <https://developer.nvidia.com/nccl>`__\ 通信库的集合通信实现。集合通信架构的典型应用方式是使用多张GPU卡协同训练，典型应用场景包括计算机视觉和自然语言处理等。

.. image:: https://github.com/PaddlePaddle/FleetX/blob/develop/docs/source/paddle_fleet_rst/collective/img/collective_arch.png?raw=true
  :width: 200
  :alt: Collective Architecture
  :align: center

典型应用场景下，如数据并行模式下，数据集也是切分到各个计算节点，每个计算节点中包含完整的模型参数，并根据本地数据训练模型，并得到本地梯度，随后所有计算节点使用集合通信原语获取全局梯度，并更新参数。

教程内容结构
~~~~~~~~~~~~~~~

- \ `Collective数据并行 <./data_parallel.rst>`\ 部分介绍Collective架构下最常用的数据并行功能。
- \ `参数服务器基础指南 <./ps.rst>`\ 部分介绍参数服务器架构的基础使用方法。
- \ `launch组件详解 <./launch.rst>`\ 部分介绍飞桨分布式训练启动组件\ `launch`\ 的使用方法。
