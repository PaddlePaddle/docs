.. _cn_api_paddle_distributed_Replicate:

Replicate
-------------------------------

.. py:class:: paddle.distributed.Replicate

描述 Tensor 在多设备间做张量的复制，即每张卡上有完全相同的张量。


代码示例
:::::::::

    .. code-block:: python

        import paddle
        import paddle.distributed as dist
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        a = paddle.ones([10, 20])
        d_tensor = dist.shard_tensor(a, mesh, [dist.Replicate()])
        print(d_tensor)
