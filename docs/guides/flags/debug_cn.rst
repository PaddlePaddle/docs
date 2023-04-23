
调试
==================


FLAGS_check_nan_inf
********************
(始于 0.13.0)

用于调试。它用于检查 Operator 的结果是否含有 Nan 或 Inf。

取值范围
---------------
Bool 型，缺省值为 False。

示例
-------
FLAGS_check_nan_inf=True - 检查 Operator 的结果是否含有 Nan 或 Inf。


FLAGS_cpu_deterministic
*******************************************
(始于 0.15.0)

该 flag 用于调试。它表示是否在 CPU 侧确定计算结果。 在某些情况下，不同求和次序的结果可能不同，例如，`a+b+c+d` 的结果可能与 `c+a+b+d` 的结果不同。

取值范围
---------------
Bool 型，缺省值为 False。

示例
-------
FLAGS_cpu_deterministic=True - 在 CPU 侧确定计算结果。


FLAGS_enable_rpc_profiler
*******************************************
(始于 1.0.0)

是否启用 RPC 分析器。

取值范围
----------------
Bool 型，缺省值为 False。

示例
-------
FLAGS_enable_rpc_profiler=True - 启用 RPC 分析器并在分析器文件中记录时间线。


FLAGS_multiple_of_cupti_buffer_size
*******************************************
(始于 1.4.0)

该 flag 用于分析。它表示 CUPTI 设备缓冲区大小的倍数。如果在 profiler 过程中程序挂掉或者在 chrome://tracing 中加载 timeline 文件时出现异常，请尝试增大此值。

取值范围
---------------
Int32 型，缺省值为 1。

示例
-------
FLAGS_multiple_of_cupti_buffer_size=1 - 将 CUPTI 设备缓冲区大小的倍数设为 1。


FLAGS_reader_queue_speed_test_mode
*******************************************
(始于 1.1.0)

将 pyreader 数据队列设置为测试模式。在测试模式下，pyreader 将缓存一些数据，然后执行器将读取缓存的数据，因此阅读器不会成为瓶颈。

取值范围
---------------
Bool 型，缺省值为 False。

示例
-------
FLAGS_reader_queue_speed_test_mode=True - 启用 pyreader 测试模式。

注意
-------
仅当使用 py_reader 时该 flag 才有效。

..  toctree::
    :hidden:

    check_nan_inf_cn.md
