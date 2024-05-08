.. _cn_api_distributed_rpc_rpc_async:

rpc_async
-------------------------------


.. py:function:: paddle.distributed.rpc.rpc_async(to, fn, args=None, kwargs=None, timeout=-1)

发起一个非阻塞的 RPC 调用，在 to 上运行函数 fn。注意：请用户务必在安全的网络环境下使用本功能。

参数
:::::::::
    - **to** (str) - 目标 worker 的名字。
    - **fn** (fn) - 一个可调用的函数，比如 Python 的函数。
    - **args** (tuple，可选) - 函数 fn 的参数，默认为 None。
    - **kwargs** (str，可选) - 函数 fn 的字典参数，默认是 None。
    - **timeout** (int，可选) - RPC 调用的超时时间，使用秒表示。如果该 RPC 调用没有在此时间内完成，则会引发异常，表示 RPC 调用超时。该值小于或等于 0 表示无限大的超时时间，即永远不会引发超时异常。默认为 -1。

返回
:::::::::
返回一个 FutureWrapper 对象。当 RPC 调用完成，fn 的运行结果可以使用 fut.wait() 的方式获取。

代码示例
:::::::::
COPY-FROM: paddle.distributed.rpc.rpc_async
