.. _cn_api_paddle_distributed_Placement:

Placement
-------------------------------

.. py:class:: paddle.distributed.Placement

描述 Tensor 分布式切分的基类，通常使用它的三个子类。请参考 :ref:`cn_api_paddle_distributed_Replicate` 、:ref:`cn_api_paddle_distributed_Shard` 、:ref:`cn_api_paddle_distributed_Partial`


代码示例
:::::::::

    .. code-block:: python

        import paddle.distributed as dist
        placements = [dist.Replicate(), dist.Shard(0), dist.Partial()]
        for p in placements:
            if isinstance(p, dist.Placement):
                if p.is_replicated():
                    print("replicate.")
                elif p.is_shard():
                    print("shard.")
                elif p.is_partial():
                    print("partial.")
