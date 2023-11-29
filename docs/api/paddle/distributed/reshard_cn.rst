.. _cn_api_paddle_distributed_reshard:

reshard
-------------------------------

.. py:function:: paddle.distributed.reshard(dist_tensor, mesh, placements)

根据新的分布式信息 ``dist_attr`` ，对一个带有分布式信息的 Tensor 进行 reshard 操作，重新进行 Tensor 的分布/切片，返回一个新的分布式 Tensor 。

``dist_tensor`` 需要是一个具有分布式信息的 paddle\.Tensor。


参数
:::::::::

    - **dist_tensor** (Tensor) - 具有分布式信息的 Tensor ，为 paddle\.Tensor 类型。
    - **mesh** (paddle.distributed.ProcessMesh) - 表示进程拓扑信息的 ProcessMesh 对象。
    - **placements** (list(Placement)) - 分布式 Tensor 的切分表示列表，描述 Tensor 在 mesh 上如何切分。

返回
:::::::::
将输入的 dist_tensor 按照新的方式进行分布/切分的分布式 Tensor。


代码示例
:::::::::

COPY-FROM: paddle.distributed.reshard
