.. _cn_api_paddle_distributed_rpc_get_worker_info:

get_worker_info
-------------------------------


.. py:function:: paddle.distributed.rpc.get_worker_info(name)

利用 worker 名字获取 worker 的信息。注意：所有 RPC 对外开放接口，均应在安全的网络环境下供内部使用，不得公网开放访问。

参数
:::::::::
    - **name** (str) - worker 的名字。

返回
:::::::::
WorkerInfo 对象，拥有属性 name，rank，ip，port。

代码示例
:::::::::
COPY-FROM: paddle.distributed.rpc.get_worker_info
