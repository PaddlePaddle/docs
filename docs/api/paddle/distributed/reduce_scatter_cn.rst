.. _cn_api_paddle_distributed_reduce_scatter:

reduce_scatter
-------------------------------


.. py:function:: paddle.distributed.reduce_scatter(tensor, tensor_list, op=ReduceOp.SUM, group=None, sync_op=True)

规约一组 tensor，随后将规约结果分发到每个进程。

.. note::
  该 API 只支持动态图模式。

参数
:::::::::
    - **tensor** (Tensor) – 用于接收数据的 tensor，数据类型必须与输入的 tensor 列表保持一致。
    - **tensor_list** (List[Tensor]) – 将被规约和分发的 tensor 列表。支持的数据类型包括：float16、float32、float64、int32、int64、int8、uint8、bool、bfloat16。
    - **op** (ReduceOp.SUM|ReduceOp.MAX|ReduceOp.MIN|ReduceOp.PROD，可选) - 归约的操作类型，包括求和、取最大值、取最小值和求乘积。默认为求和。
    - **group** (Group，可选) - 执行该操作的进程组实例（通过 ``new_group`` 创建）。默认为 None，即使用全局默认进程组。
    - **sync_op** (bool，可选) - 该操作是否为同步操作。默认为 True，即同步操作。


返回
:::::::::
若为同步操作，无返回值；若为异步操作，返回 ``Task``。通过 ``Task``，可以查看异步操作的执行状态以及等待异步操作的结果。

代码示例
:::::::::
COPY-FROM: paddle.distributed.reduce_scatter
