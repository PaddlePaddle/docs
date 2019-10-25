
调试
==================


FLAGS_check_nan_inf
********************
(始于0.13.0)

用于调试。它用于检查Operator的结果是否含有Nan或Inf。

取值范围
---------------
Bool型，缺省值为False。

示例
-------
FLAGS_check_nan_inf=True - 检查Operator的结果是否含有Nan或Inf。


FLAGS_cpu_deterministic
*******************************************
(始于0.15.0)

该flag用于调试。它表示是否在CPU侧确定计算结果。 在某些情况下，不同求和次序的结果可能不同，例如，`a+b+c+d` 的结果可能与 `c+a+b+d` 的结果不同。

取值范围
---------------
Bool型，缺省值为False。

示例
-------
FLAGS_cpu_deterministic=True - 在CPU侧确定计算结果。


FLAGS_enable_rpc_profiler
*******************************************
(始于1.0.0)

是否启用RPC分析器。

取值范围
----------------
Bool型，缺省值为False。

示例
-------
FLAGS_enable_rpc_profiler=True - 启用RPC分析器并在分析器文件中记录时间线。


FLAGS_multiple_of_cupti_buffer_size
*******************************************
(始于1.4.0)

该flag用于分析。它表示CUPTI设备缓冲区大小的倍数。如果在profiler过程中程序挂掉或者在chrome://tracing中加载timeline文件时出现异常，请尝试增大此值。

取值范围
---------------
Int32型，缺省值为1。

示例
-------
FLAGS_multiple_of_cupti_buffer_size=1 - 将CUPTI设备缓冲区大小的倍数设为1。


FLAGS_reader_queue_speed_test_mode
*******************************************
(始于1.1.0)

将pyreader数据队列设置为测试模式。在测试模式下，pyreader将缓存一些数据，然后执行器将读取缓存的数据，因此阅读器不会成为瓶颈。

取值范围
---------------
Bool型，缺省值为False。

示例
-------
FLAGS_reader_queue_speed_test_mode=True - 启用pyreader测试模式。

注意
-------
仅当使用py_reader时该flag才有效。