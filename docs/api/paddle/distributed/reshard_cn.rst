.. _cn_api_paddle_distributed_reshard:

reshard
-------------------------------

.. py:function:: paddle.distributed.reshard(dist_tensor, dist_attr)

根据新的分布式信息 ``dist_attr`` ，对一个带有分布式信息的 Tensor 进行 reshard 操作，重新进行 Tensor 的分布/切片，返回一个新的分布式 Tensor 。

``dist_tensor`` 需要是一个具有分布式信息的 paddle\.Tensor。


参数
:::::::::

    - **dist_tensor** (Tensor) - 具有分布式信息的 Tensor ，为 paddle\.Tensor 类型。
    - **dist_attr** (paddle.distributed.DistAttr) - Tensor 在 ProcessMesh 上的新的分布/切片方式。

返回
:::::::::
将输入的 dist_tensor 按照新的方式进行分布/切分的分布式 Tensor。


代码示例
:::::::::

COPY-FROM: paddle.distributed.reshard
