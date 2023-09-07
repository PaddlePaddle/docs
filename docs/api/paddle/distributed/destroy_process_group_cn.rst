.. _cn_api_paddle_distributed_destroy_process_group:

destroy_process_group
-------------------------------


.. py:function:: destroy_process_group(group=None)

销毁一个指定的通信组。

参数
:::::::::
    - group (ProcessGroup, 可选): 待销毁的通信组。所有通信组都会被销毁（包括默认的通信组），并且整个分布式环境也会回到未被初始化的状态。

返回
:::::::::
无返回值。

代码示例
::::::::::::
COPY-FROM: paddle.distributed.destroy_process_group
