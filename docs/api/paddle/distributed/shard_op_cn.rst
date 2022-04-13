.. _cn_api_distributed_shard_op:

shard_op
-------------------------------

.. py:function:: paddle.distributed.shard_op(op_fn, dist_attr=None)

调用函数，并为该函数所增加的op添加分布式属性。

参数
:::::::::
    - op_fn (callable) - 待切分的可调用算子或模块。
    - dist_attr (dict) - 算子分布式属性。将接受的属性分为两类。第一类描述了为所有输入和输出
        共享的分布式属性，现在只能指定process_mesh。第二类描述的是分布式输入或输出的属性与
        shard_tensor的dist_attr相同。这两类都是可选的，用户可以根据需要指定它们。   
        注意，算子的process_mesh必须是与这些用于输入和输出的process_mesh相同。

返回
:::::::::
列表:函数 `op_fn` 的输出，用分布式属性标注。

代码示例
:::::::::
.. code-block:: python

    import paddle
    import paddle.distributed as dist

    paddle.enable_static()

    x = paddle.ones([4, 6])
    y = paddle.zeros([4, 6])
    dist_add = dist.shard_op(paddle.add,
                                dist_attr={
                                    "process_mesh": [[2, 3, 1], [0, 4, 5]],
                                    x: {"dims_mapping": [-1, 0]},
                                    y: {"dims_mapping": [0, -1]}
                                })
    dist_add(x, y)