.. _cn_api_distributed_get_rank:

get_rank
----------

..  py:function:: paddle.distributed.get_rank()

返回当前进程的 rank。

当前进程 rank 的值等于环境变量 ``PADDLE_TRAINER_ID`` 的值，默认值为 0。

返回
:::::::::
(int) 当前进程的 rank。

代码示例
:::::::::

COPY-FROM: paddle.distributed.get_rank
