.. _cn_api_paddle_distributed_ReduceType:

ReduceType
-------------------------------

.. py:class:: paddle.distributed.ReduceType

指定分布式 Tensor 在 Partial 状态下规约操作的类型，必须是下述值之一：

    ReduceType.kRedSum

    ReduceType.kRedMax

    ReduceType.kRedMin

    ReduceType.kRedProd

    ReduceType.kRedAvg

    ReduceType.kRedAny

    ReduceType.kRedAll

代码示例
:::::::::

    .. code-block:: python

        import paddle
        import paddle.distributed as dist
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        a = paddle.ones([10, 20])
        d_tensor = dist.shard_tensor(a, mesh, [dist.Partial(dist.ReduceType.kRedSum)])
        print(d_tensor)
