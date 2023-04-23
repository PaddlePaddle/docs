.. _cn_api_distributed_get_rank:

get_rank
----------

..  py:function:: paddle.distributed.get_rank(group=None)

返回当前进程在给定通信组下的 rank，rank 是在 [0, world_size) 范围内的连续整数。如果没有指定通信组，则默认使用全局通信组。

参数
:::::::::
    - **group** (Group，可选) - 指定想在哪个通信组下得到当前进程的 rank，如果没有指定，默认使用全局通信组。

返回
:::::::::
(int) 返回当前进程在指定通信组中的 rank，如果当前进程不在该通信组中，则返回-1。

代码示例
:::::::::

COPY-FROM: paddle.distributed.get_rank
