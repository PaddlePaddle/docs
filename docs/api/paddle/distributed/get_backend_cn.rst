.. _cn_api_distributed_get_backend:

get_backend
-------------------------------

.. py:function:: paddle.distributed.get_backend(group=None)

获取指定分布式通信组后端的名称。

参数
:::::::::
    - **group** (Group，可选) - 指定的通信组。默认为 None，即使用全局默认进程组。

返回
:::::::::
``str``，通信组后端的名称。

代码示例
::::::::::::
COPY-FROM: paddle.distributed.get_backend
