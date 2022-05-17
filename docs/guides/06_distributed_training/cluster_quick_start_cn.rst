..  _cluster_quick_start:

Paddle分布式整体介绍
====================================

概述
------

分布式训练是在单机算力和存储能力有限的情况下，通过多计算节点协同工作，以实现训练加速或者解决单机无法训练的问题。大规模的数据可以让模型有足够的“教材”用于“学习”，而大规模的参数量则可以让模型“学习能力”更强，更容易 “学习” 到“教材”中的“知识”。在数据和参数规模增长的过程中，常规的单机训练由于硬件资源的限制渐渐显得捉襟见肘，而分布式训练则成为了广大开发者的必然选择。这其中涉及多机任务拆分（多种并行策略）、集群训练资源配置、平衡训练速度和收敛速度、弹性训练与容错等多项重要技术。

PaddleFleet是Paddle的分布式模块，其功能放置在了`Paddle.distributed`这个Package下。在这篇文章中，我们将会整体介绍一下这个Pacakge的各个组成部分以便让您对其有一个整体的感知。

.. note:: 注：分布式功能从不同的维度可能会有不同的分类方式。比如通信模式角度，动静角度，功能角度等等。考虑到用户实际使用关心的焦点不同，我们这里按照功能角度进行分类的介绍

* `数据并行训练 <https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/collective/data_parallel.html>`__ 数据并行训练是分布式并行训练中用的最早、也是最广泛的一个功能。和混合并行训练不同，它适合于模型单卡能放下，多卡间复制模型（SPMD:single program, multiple data）然后同步通信参数的梯度、提高minibatchsize的方式提高训练效率。


* `混合并行训练 <https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/collective/collective_mp/hybrid_parallelism.html>`__ 更大的模型一般会有更好的效果，但是更大的模型会带来显存瓶颈、计算瓶颈、通信扩展瓶颈的问题，为了更好的解决这些问题，我们提出了高效的 `4D混合并行的方式 <https://ai.baidu.com/forum/topic/show/987996>`__ ，即 `数据并行 <https://>`__ 、`流水线并行 <https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/collective/collective_mp/pipeline.html>`__ 、`张量并行 <https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/collective/collective_mp/model_parallel.html>`__、`GroupSharding <https://>`__，指南里面描述了详细的代码结构骨架和完整代码连接)并行，可以让用户根据模型的规模和机器资源规模来选择不同的并行方式来进行高效的训练。PaddleFleet通过API方式提供了4D混合并行的方法，这些功能是垂直的、可以相互嵌套使用的。

* `MoE并行训练 <https://>`__ 与Dense大模型不同，MoE训练过程中只会激活部分的Expert参数从而大幅减少了计算量。目前MoE成为了通往万亿以及更大的模型的主要方式。优于模型规模和计算资源规模的提升，MoE训练仍然面临计算瓶颈和通信瓶颈。PaddleMoE并行训练提供了业内高效的MoE训练常用的[`Gate` 和 `MoELayer`]()方式。

* `自动并行训练 <https://>`__ 自动并行能根据用户输入串行网络模型和集群资源信息自动进行分布式训练，支持半自动与全自动两种模式，半自动模式下用户可以指定某些tensor和operator的切分方式，而全自动模式下所有tensor和operator都由框架自适应选择最优切分策略。

* `PS-based并行训练 <https://>`__ 参数服务器由高性能异步训练 Worker、高效通信策略和高性能 Server 组成。Server节点负责参数的创建，更新和保存。Worker负责训练数据IO，模型前向反向计算。Server和Worker的通信主要包括参数从Server拉取以及更新梯度到Server。Worker数据并行的方式能够激发多Worker节点的吞吐量优势。在异步训练模式下训练简单模型可以极大提升数据吞吐量，Server分片存储机制能够支持超大模型规模。

* `通信模块 <https://>`__ 一般情况下，用户不需要直接调用通信模块的API，因为各种并行的方式已经集成了通信部分。当您需要[比如PipeLineParallel的自定义组网](此处连接到PipeLine的Send/Recv部分)时，或者各个训练进程之间需要显式的通信比如loss、配置文件等内容时，可以使用通信模块所提供的collective和P2P的通信接口。

* `弹性训练 <https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/edl.html>`__ 弹性训练提供两方面的能力，任务所需要的资源可以随训练进度变化，以及当所能分配给任务的资源变化时对任务进行动态调整，前者可以保证任务能够充分利用可用资源提高训练效率，后者可以提高集群的资源利用率。PaddlePaddle 的弹性训练能够根据任务需求动态调整训练节点数和训练参数以提升训练效率，例如资源空闲时扩充训练节点加快训练进度，资源过载时收缩部分任务节点优先保证高优任务训练。

* `云端训练的支持 <https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/public_cloud.html>`__ 针对常见的云平台，我们提供了在其上运行任务的详细的方法和步骤。

