.. _cn_api_distributed_gloo_init_parallel_env:

gloo_init_parallel_env
-----------------

.. py:function:: paddle.distributed.gloo_init_parallel_env(rank_id, rank_num, server_endpoint)
该函数仅初始化 ``GLOO`` 上下文用于 CPU 间的通信。

参数
:::::::::
    - **rank_id** (int) - 当前 rank 的编号；
    - **rank_num** (int) - 当前并行环境中 rank 的总数；
    - **server_endpoint** (str) - 用于初始化 gloo 上下文的服务器端点，格式为 ip:port。

返回
:::::::::
无

代码示例
:::::::::
COPY-FROM: paddle.distributed.gloo_init_parallel_env
