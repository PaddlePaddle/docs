.. _cn_api_distributed_rpc_shutdown:

shutdown
-------------------------------


.. py:function:: paddle.distributed.rpc.shutdown()

关闭 RPC 代理和 worker。这将阻塞直到所有本地和远程 RPC 进程都达到此方法并等待所有未完成的工作完成。

参数
:::::::::
无

返回
:::::::::
无

代码示例
:::::::::
COPY-FROM: paddle.distributed.rpc.shutdown
