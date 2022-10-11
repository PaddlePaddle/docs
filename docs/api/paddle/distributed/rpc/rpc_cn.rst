.. _cn_api_distributed_rpc_rpc:

rpc
-------------------------------

init_rpc(name, rank=None, world_size=None, master_endpoint=None)
:::::::::

.. py:function:: paddle.distributed.rpc.init_rpc(name, rank=None, world_size=None, master_endpoint=None)

初始化 rpc 服务。

参数
'''''''''
    - **name** (str) - worker 名字。
    - **rank** (int，可选) - worker 的 ID，默认为 None。
    - **world_size** (int，可选) - workers 的数量，默认为 None。
    - **master_endpoint** (str，可选) - master 的 IP 地址，其他 worker 节点通过 master 来获取彼此的信息，默认是 None。

返回
'''''''''
无

代码示例
'''''''''
COPY-FROM: paddle.distributed.rpc.init_rpc

rpc_sync(to, fn, args=None, kwargs=None, timeout=_DEFAULT_RPC_TIMEOUT)
:::::::::

.. py:function:: paddle.distributed.rpc.rpc_sync(to, fn, args=None, kwargs=None, timeout=_DEFAULT_RPC_TIMEOUT)

发起一个阻塞的 RPC 调用，在 to 上运行函数 fn

参数
'''''''''
    - **to** (str) - 目标 worker 的名字。
    - **fn** (fn) - 一个可调用的函数，比如 Python 的函数。
    - **args** (tuple，可选) - 函数 fn 的参数，默认为 None。
    - **kwargs** (str，可选) - 函数 fn 的字典参数，默认是 None。
    - **timeout** (int，可选) - RPC 调用的超时时间，使用秒表示。如果该 RPC 调用没有在此时间内完成，则会引发异常，表示 RPC 调用超时。该值小于或等于 0 表示无限大的超时时间，即永远不会引发超时异常。默认为 -1。

返回
'''''''''
返回以 args 和 kwargs 作为参数的函数 fn 的运行结果

代码示例
'''''''''
COPY-FROM: paddle.distributed.rpc.rpc_sync

rpc_async(to, fn, args=None, kwargs=None, timeout=_DEFAULT_RPC_TIMEOUT)
:::::::::

.. py:function:: paddle.distributed.rpc.rpc_async(to, fn, args=None, kwargs=None, timeout=_DEFAULT_RPC_TIMEOUT)

发起一个非阻塞的 RPC 调用，在 to 上运行函数 fn

参数
'''''''''
    - **to** (str) - 目标 worker 的名字。
    - **fn** (fn) - 一个可调用的函数，比如 Python 的函数。
    - **args** (tuple，可选) - 函数 fn 的参数，默认为 None。
    - **kwargs** (str，可选) - 函数 fn 的字典参数，默认是 None。
    - **timeout** (int，可选) - RPC 调用的超时时间，使用秒表示。如果该 RPC 调用没有在此时间内完成，则会引发异常，表示 RPC 调用超时。该值小于或等于 0 表示无限大的超时时间，即永远不会引发超时异常。默认为 -1。

返回
'''''''''
返回一个 FutureWrapper 对象。当 RPC 调用完成，fn 的运行结果可以使用 fut.wait() 的方式获取。

代码示例
'''''''''
COPY-FROM: paddle.distributed.rpc.rpc_async

shutdown()
:::::::::

.. py:function:: paddle.distributed.rpc.shutdown()

关闭 RPC 代理和 worker。这将阻塞直到所有本地和远程 RPC 进程都达到此方法并等待所有未完成的工作完成。

参数
'''''''''
无

返回
'''''''''
无

代码示例
'''''''''
COPY-FROM: paddle.distributed.rpc.shutdown

get_worker_info(name)
:::::::::

.. py:function:: paddle.distributed.rpc.get_worker_info(name)

利用 worker 名字获取 worker 的信息。

参数
'''''''''
    - **name** (str) - worker 的名字。

返回
'''''''''
WorkerInfo 对象，拥有属性 name，rank，ip，port。

代码示例
'''''''''
COPY-FROM: paddle.distributed.rpc.get_worker_info

get_all_worker_infos()
:::::::::
.. py:function:: paddle.distributed.rpc.get_all_worker_infos()

获取所有 worker 的信息。

参数
'''''''''
无

返回
'''''''''
List[WorkerInfo]

代码示例
'''''''''
COPY-FROM: paddle.distributed.rpc.get_all_worker_infos

get_current_worker_info()
:::::::::
.. py:function:: paddle.distributed.rpc.get_current_worker_info()

获取当前 worker 的信息。

参数
'''''''''
无

返回
'''''''''
WorkerInfo 对象，拥有属性 name，rank，ip，port。

代码示例
'''''''''
COPY-FROM: paddle.distributed.rpc.get_current_worker_info
