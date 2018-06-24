# Fluid分布式训练手册

## 分布式训练基本思想

分布式深度学习训练通常分为两种并行化方法：数据并行，模型并行，参考下图：

<img src="parallelism.png">

在模型并行方式下，模型的层和参数将被分布在多个节点上，模型在一个mini-batch的前向和反向训练中，将经过多次跨
节点之间的通信。每个节点只保存整个模型的一部分；在数据并行方式下，每个节点保存有完整的模型的层和参数，每个节点
独自完成前向和反向计算，然后完成梯度的聚合并同步的更新所有节点上的参数。Fluid目前版本仅提供数据并行方式，另外
诸如模型并行的特例实现（超大稀疏模型训练）功能将在后续的文档中予以说明。

在数据并行模式的训练中，Fluid使用了两种通信模式，用于应对不同训练任务对分布式训练的要求，分别为RPC通信和Collective
通信。其中RPC通信方式使用[gRPC](https://github.com/grpc/grpc/)，Collective通信方式使用
[NCCL2](https://developer.nvidia.com/nccl)。下面是一个RPC通信和Collective通信的横向对比：

| Feature       | Collective    | RPC   |
| ------------- |:-------------:| -----:|
| Ring-Based Comm  | Yes | No |
| Async Training   | Reduce ranks | Fast, Direct async updates |
| Dist-Sparse-Table | No      | Yes |
| Fault-Tolerant | No | Yes|
| Performance | Faster | Fast |

* RPC通信方式的结构：
  <img src="">
* NCCL2通信方式的结构：
  <img src="">


## 使用parameter server方式的训练


## 使用NCCL2通信方式的训练


