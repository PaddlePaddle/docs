.. _cn_api_distributed_shard_op:

shard_op
-------------------------------


.. py:function:: paddle.distributed.shard_op(op_fn, mesh, dims_mapping_dict, **kwargs)

调用某个函数，并为该函数添加的操作算子设置分布式属性。

参数
:::::::::
    - op_fn (callable) - 某个API的可调用对象。
    - mesh (ProcessMesh) - ProcessMesh实例，指定该函数调用添加的操作算子的逻辑进程拓扑信息。
    - dim_mapping_dict (dict) - Tensor的名字和其对应的dim_mapping结构的映射表。其中，dim_mapping是描述某个Tensor和 `mesh` 之间映射关系的列表，该Tensor的维度 `i` 沿mesh的维度 `dim_mapping[i]` 切分，值-1表示不切分。
    - kwargs (dict) - 传递给 `op_fn` 的参数字典。

返回
:::::::::
list: 函数 `op_fn` 的输出列表。

代码示例
:::::::::
.. code-block:: python

    import paddle
    import paddle.distributed as dist

    paddle.enable_static()

    mesh = dist.ProcessMesh([[2, 4, 5], [0, 1, 3]])
    x = paddle.ones([4, 6])
    y = paddle.zeros([4, 6])
    kwargs = {'x': x, 'y': y}
    dist.shard_op(paddle.add, mesh, None, **kwargs)
