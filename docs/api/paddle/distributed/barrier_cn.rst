.. _cn_api_distributed_barrier:

barrier
-------------------------------


.. py:function:: paddle.distributed.barrier(group=0)

同步进程组内的所有进程。

参数
:::::::::
    - **group** (int，可选) - 工作的进程组编号，默认为 0。

返回
:::::::::
无

代码示例
:::::::::
COPY-FROM: paddle.distributed.barrier
