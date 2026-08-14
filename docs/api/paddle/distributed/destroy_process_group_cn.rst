.. _cn_api_paddle_distributed_destroy_process_group:

destroy_process_group
-------------------------------


.. py:function:: paddle.distributed.destroy_process_group(group=None)

销毁一个指定的通信组。

参数
:::::::::
    - **group** (Group，可选): 待销毁的通信组。未指定或指定全局默认通信组时，所有通信组（包括默认通信组）都会被销毁，整个分布式环境会回到未初始化状态；指定非全局通信组时，仅销毁该通信组。

返回
:::::::::
无返回值。

代码示例
::::::::::::
COPY-FROM: paddle.distributed.destroy_process_group
