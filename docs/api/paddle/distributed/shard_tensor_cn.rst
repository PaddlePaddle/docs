.. _cn_api_distributed_shard_tensor:

shard_tensor
-------------------------------


.. py:function:: paddle.distributed.shard_tensor(x, mesh, dim_mapping)

为输入Tensor `x` 设置分布式属性。

参数
:::::::::
    - x (Tensor) - 待处理的输入Tensor。
    - mesh (ProcessMesh) - 描述逻辑进程拓扑信息的ProcessMesh实例。
    - dim_mapping (list) - 描述`x`和`mesh`之间映射关系的列表。`x`的第`i`维沿着`mesh`的第`dim_mapping[i]`维切分。值-1表示不沿着该维切分。

返回
:::::::::
Tensor: 输入`x`自身。

代码示例
:::::::::
.. code-block:: python

    import paddle
    import paddle.distributed as dist

    paddle.enable_static()

    mesh = dist.ProcessMesh([[2, 4, 5], [0, 1, 3]])
    x = paddle.ones([4, 6])
    dist.shard_tensor(x, mesh, [0, -1])
