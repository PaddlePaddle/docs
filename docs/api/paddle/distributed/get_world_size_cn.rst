.. _cn_api_paddle_distributed_get_world_size:

get_world_size
----------------

.. py:function:: paddle.distributed.get_world_size(group=None)

返回指定通信组中的进程数，如果没有指定通信组，则默认使用全局通信组。

参数
:::::::::
    - **group** (Group，可选) - 指定想在得到哪个通信组下的进程数，如果没有指定，默认使用全局通信组。

返回
:::::::::
(int) 返回指定通信组中的进程数，如果当前进程不在该通信组中，则返回-1。

代码示例
:::::::::
COPY-FROM: paddle.distributed.get_world_size
