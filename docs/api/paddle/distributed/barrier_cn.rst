.. _cn_api_paddle_distributed_barrier:

barrier
-------------------------------


.. py:function:: paddle.distributed.barrier(group=None)

同步进程组内的所有进程。

参数
:::::::::
    - **group** (Group，可选) - 执行该操作的进程组实例（通过 ``new_group`` 创建）。默认为 None，即使用全局默认进程组。

返回
:::::::::
无返回值。

代码示例
:::::::::
COPY-FROM: paddle.distributed.barrier
