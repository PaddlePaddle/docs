.. _cn_api_paddle_distributed_new_group:

new_group
-------------------------------


.. py:function:: new_group(ranks=None, backend=None)

创建分布式通信组。


参数
:::::::::
    - **ranks** (list) - 用于新建通信组的全局 rank 列表
    - **backend** (str) - 用于新建通信组的后端支持，目前仅支持 nccl


返回
:::::::::
Group：新建的通信组对象

代码示例
::::::::::::
COPY-FROM: paddle.distributed.new_group
