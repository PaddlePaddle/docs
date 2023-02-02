.. _cn_api_distributed_spawn:

spawn
-----

.. py:function:: paddle.distributed.spawn(func, args=(), nprocs=-1, join=True, daemon=False, **options)

使用 ``spawn`` 方法启动多进程任务。

.. note::
    ``spawn`` 目前仅支持 GPU 和 XPU 的 collective 模式。GPU 和 XPU 的 collective 模式不能同时启动，因此 `gpus` 和 `xpus` 这两个参数不能同时配置。

参数
:::::::::
    - **func** (function) - 由 ``spawn`` 方法启动的进程所调用的目标函数。该目标函数需要能够被 ``pickled`` (序列化)，所以目标函数必须定义为模块的一级函数，不能是内部子函数或者类方法。
    - **args** (tuple，可选) - 传入目标函数 ``func`` 的参数。
    - **nprocs** (int，可选) - 启动进程的数目。默认值为-1。当 ``nproc`` 为-1 时，模型执行时将会从环境变量中获取当前可用的所有设备进行使用：如果使用 GPU 执行任务，将会从环境变量 ``CUDA_VISIBLE_DEVICES`` 中获取当前所有可用的设备 ID；如果使用 XPU 执行任务，将会从环境变量 ``XPU_VISIBLE_DEVICES`` 中获取当前所有可用的设备 ID。
    - **join** (bool，可选) - 对所有启动的进程执行阻塞的 ``join``，等待进程执行结束。默认为 True。
    - **daemon** (bool，可选) - 配置启动进程的 ``daemon`` 属性。默认为 False。
    - **options (dict，可选) - 其他初始化并行执行环境的配置选项。目前支持以下选项：(1) start_method (string) - 启动子进程的方法。进程的启动方法可以是 ``spawn`` ， ``fork`` , ``forkserver``。因为 CUDA 运行时环境不支持 ``fork`` 方法，当在子进程中使用 CUDA 时，需要使用 ``spawn`` 或者 ``forkserver`` 方法启动进程。默认方法为 ``spawn`` ； (2) gpus (string) - 指定训练使用的 GPU ID，例如 "0,1,2,3"，默认值为 None ； (3) xpus (string) - 指定训练使用的 XPU ID，例如 "0,1,2,3"，默认值为 None ； (4) ips (string) - 运行集群的节点（机器）IP，例如 "192.168.0.16,192.168.0.17"，默认值为 "127.0.0.1" 。

返回
:::::::::
 ``MultiprocessContext`` 对象，持有创建的多个进程。

代码示例
:::::::::
COPY-FROM: paddle.distributed.spawn
