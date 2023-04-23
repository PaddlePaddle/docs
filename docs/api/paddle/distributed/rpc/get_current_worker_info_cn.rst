.. _cn_api_distributed_rpc_get_current_worker_info:

get_current_worker_info
-------------------------------


.. py:function:: paddle.distributed.rpc.get_current_worker_info()

获取当前 worker 的信息。

参数
:::::::::
无

返回
:::::::::
WorkerInfo 对象，拥有属性 name，rank，ip，port。

代码示例
:::::::::
COPY-FROM: paddle.distributed.rpc.get_current_worker_info
