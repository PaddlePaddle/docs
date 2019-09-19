
其他
==================



FLAGS_benchmark
********************
(始于0.12.0)

用于基准测试。设置后，它将使局域删除同步，添加一些内存使用日志，并在内核启动后同步所有cuda内核。

取值范围
---------------
Bool型，缺省值为False。

示例
-------
FLAGS_benchmark=True -  同步以测试基准。


FLAGS_inner_op_parallelism
*******************************************
(始于1.3.0)

大多数Operators都在单线程模式下工作，但对于某些Operators，使用多线程更合适。 例如，优化稀疏梯度的优化Op使用多线程工作会更快。该flag用于设置Op内的线程数。

取值范围
---------------
Int32型，缺省值为0，这意味着operator将不会在多线程模式下运行。

示例
-------
FLAGS_inner_op_parallelism=5 - 将operator内的线程数设为5。

注意
-------
目前只有稀疏的adam op支持inner_op_parallelism。


FLAGS_max_body_size
*******************************************
(始于1.0.0)

控制BRPC中的最大消息大小。

取值范围
---------------
Int32型，缺省值为2147483647。

示例
-------
FLAGS_max_body_size=2147483647 - 将BRPC消息大小设为2147483647。


FLAGS_sync_nccl_allreduce
*******************************************
(始于1.3)

如果FLAGS_sync_nccl_allreduce为True，则会在allreduce_op_handle中调用 `cudaStreamSynchronize（nccl_stream）` ，这种模式在某些情况下可以获得更好的性能。

取值范围
---------------
Bool型，缺省值为True。

示例
-------
FLAGS_sync_nccl_allreduce=True - 在allreduce_op_handle中调用 `cudaStreamSynchronize(nccl_stream)` 。


FLAGS_tracer_profile_fname
*******************************************
(始于1.4.0)

FLAGS_tracer_profile_fname表示由gperftools生成的命令式跟踪器的分析器文件名。仅在编译选项选择`WITH_PROFILER = ON`时有效。如果禁用则设为empty。

取值范围
---------------
String型，缺省值为("gperf")。

示例
-------
FLAGS_tracer_profile_fname="gperf_profile_file" - 将命令式跟踪器的分析器文件名设为"gperf_profile_file"。

