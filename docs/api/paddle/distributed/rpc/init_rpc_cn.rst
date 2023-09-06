.. _cn_api_distributed_rpc_init_rpc:

init_rpc
-------------------------------


.. py:function:: paddle.distributed.rpc.init_rpc(name, rank=None, world_size=None, master_endpoint=None)

初始化 rpc 服务。

参数
:::::::::
    - **name** (str) - worker 名字。
    - **rank** (int，可选) - worker 的 ID，默认为 None。
    - **world_size** (int，可选) - workers 的数量，默认为 None。
    - **master_endpoint** (str，可选) - master 的 IP 地址，其他 worker 节点通过 master 来获取彼此的信息，默认是 None。

返回
:::::::::
无

代码示例
:::::::::
COPY-FROM: paddle.distributed.rpc.init_rpc
