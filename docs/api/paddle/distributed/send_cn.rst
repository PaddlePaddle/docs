.. _cn_api_distributed_send:

send
-------------------------------


.. py:function:: paddle.distributed.send(tensor, dst=0, group=None, sync_op=True)

发送一个 tensor 到指定进程。

参数
:::::::::
    - **tensor** (Tensor) - 待发送的 Tensor。支持的数据类型包括：float16、float32、float64、int32、int64、int8、uint8、bool、bfloat16。
    - **dst** (int，可选) - 目标进程的 rank，传入的 tensor 将发送到该进程。默认为 0，即发送到 rank=0 的进程。
    - **group** (Group，可选) - 执行该操作的进程组实例（通过 ``new_group`` 创建）。默认为 None，即使用全局默认进程组。
    - **sync_op** (bool，可选) - 该操作是否为同步操作。默认为 True，即同步操作。

返回
:::::::::
动态图模式下，若为同步操作，无返回值；若为异步操作，返回 ``Task``。通过 ``Task``，可以查看异步操作的执行状态以及等待异步操作的结果。

静态图模式下，无返回值。

代码示例
:::::::::
COPY-FROM: paddle.distributed.send
