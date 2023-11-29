.. _cn_api_paddle_distributed_Partial:

Partial
-------------------------------

.. py:class:: paddle.distributed.Partial

描述 Tensor 在多设备间，此类型的张量具有相同的 shape，但只有一部分数值，其可以进一步做规约（如 sum/min/max）以得到 dist_tensor，通常用做一种中间表示。


参数
:::::::::

    - **reduce_type** (paddle.distributed.ReduceType) - 在 Partial 状态下规约操作的类型, 默认 paddle.distributed.ReduceType.kRedSum。


代码示例
:::::::::

    .. code-block:: python

        import paddle
        import paddle.distributed as dist
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        a = paddle.ones([10, 20])
        d_tensor = dist.shard_tensor(a, mesh, [dist.Partial()])
        out_d_tensor = dist.reshard(d_tensor, mesh, [dist.Replicate()])
        print(out_d_tensor)
