..  _distributed_collective_faq:

1. **问题：为何单机多卡训练正常，多机训练提示连接不上或者创建连接初始化错误？**
    答：检查一下 `NCCL_SOCKET_IFNAME` 环境变量。如果机器有多个网卡，NCCL有可能在创建连接的时候选错网卡。
       一般用 `export NCCL_SOCKET_IFNAME=` 限定网卡即可

#. **问题：为何数据并行多卡保存的模型不一致？**
    答：检查一下组网中是不是有BN（BatchNorm）层。BN层保存的Mean和variance可能不一致。
       我们一般只让卡0（或者rank0）保存模型，而不是多个卡同时保存模型。

#. **问题：NCCL通信影响性能的常见环境变量都有哪些？**
    答：常见的有：`NCCL_IB_DISABLE` `NCCL_NET_GDR_LEVEL` `NCCL_P2P_LEVEL`等。
       你可以在NV的 `官方页面 <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html>`__ 中找到对应的说明。
       其中容易忽视的是 `NCCL_IB_DISABLE` 参数。如果设置 `export NCCL_IB_DISABLE=1` 会让GPU通信不使用IB协议，对速度影响比较大。