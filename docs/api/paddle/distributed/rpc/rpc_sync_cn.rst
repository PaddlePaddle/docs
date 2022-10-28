.. _cn_api_distributed_rpc_rpc_sync:

rpc_sync
-------------------------------

.. py:function:: paddle.distributed.rpc.rpc_sync(to, fn, args=None, kwargs=None, timeout=-1)

发起一个阻塞的 RPC 调用，在 to 上运行函数 fn

参数
:::::::::
    - **to** (str) - 目标 worker 的名字。
    - **fn** (fn) - 一个可调用的函数，比如 Python 的函数。
    - **args** (tuple，可选) - 函数 fn 的参数，默认为 None。
    - **kwargs** (str，可选) - 函数 fn 的字典参数，默认是 None。
    - **timeout** (int，可选) - RPC 调用的超时时间，使用秒表示。如果该 RPC 调用没有在此时间内完成，则会引发异常，表示 RPC 调用超时。该值小于或等于 0 表示无限大的超时时间，即永远不会引发超时异常。默认为 -1。

返回
:::::::::
返回以 args 和 kwargs 作为参数的函数 fn 的运行结果

代码示例
:::::::::
COPY-FROM: paddle.distributed.rpc.rpc_sync
