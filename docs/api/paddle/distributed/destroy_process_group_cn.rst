.. _cn_api_distributed_destroy_process_group:

destroy_process_group
-------------------------------


.. py:function:: paddle.distributed.destroy_process_group(group=None)

销毁指定的通信组，或取消分布式初始设置。

参数
:::::::::
    - group (Group, optional) – 指定销毁的通信组；如果为None或默认通信组，则包括默认通信组在内的所有group都会被销毁。

返回
:::::::::
无

代码示例
:::::::::
COPY-FROM: paddle.distributed.destroy_process_group