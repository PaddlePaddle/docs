.. _cn_api_paddle_distributed_Shard:

Shard
-------------------------------

.. py:class:: paddle.distributed.Shard

描述 Tensor 在多设备间按照指定的维度来切分张量。


参数
:::::::::

    - **dim** (int) - 指定张量的切分维度。


代码示例
:::::::::

    .. code-block:: python

        import paddle
        import paddle.distributed as dist
        mesh = dist.ProcessMesh([[2, 4, 5], [0, 1, 3]], dim_names=['x', 'y'])
        a = paddle.to_tensor([[1,2,3],[5,6,7]])
        d_tensor = dist.shard_tensor(a, mesh, [dist.Shard(0), dist.Shard(1)])
        print(d_tensor)
