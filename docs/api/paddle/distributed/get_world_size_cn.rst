.. _cn_api_distributed_get_world_size:

get_world_size
----------------

.. py:function:: paddle.distributed.get_world_size()

返回参与当前任务的进程数。

当前进程数等于环境变量 ``PADDLE_TRAINERS_NUM`` 的值，默认值为 1。

返回
:::::::::
(int) 参与任务的进程数。

代码示例
:::::::::
COPY-FROM: paddle.distributed.get_world_size
