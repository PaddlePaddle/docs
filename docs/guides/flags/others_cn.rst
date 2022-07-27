
其他
==================



FLAGS_benchmark
********************
(始于 0.12.0)

用于基准测试。设置后，它将使局域删除同步，添加一些内存使用日志，并在内核启动后同步所有 cuda 内核。

取值范围
---------------
Bool 型，缺省值为 False。

示例
-------
FLAGS_benchmark=True -  同步以测试基准。


FLAGS_inner_op_parallelism
*******************************************
(始于 1.3.0)

大多数 Operators 都在单线程模式下工作，但对于某些 Operators，使用多线程更合适。 例如，优化稀疏梯度的优化 Op 使用多线程工作会更快。该 flag 用于设置 Op 内的线程数。

取值范围
---------------
Int32 型，缺省值为 0，这意味着 operator 将不会在多线程模式下运行。

示例
-------
FLAGS_inner_op_parallelism=5 - 将 operator 内的线程数设为 5。

注意
-------
目前只有稀疏的 adam op 支持 inner_op_parallelism。


FLAGS_max_body_size
*******************************************
(始于 1.0.0)

控制 BRPC 中的最大消息大小。

取值范围
---------------
Int32 型，缺省值为 2147483647。

示例
-------
FLAGS_max_body_size=2147483647 - 将 BRPC 消息大小设为 2147483647。


FLAGS_sync_nccl_allreduce
*******************************************
(始于 1.3)

如果 FLAGS_sync_nccl_allreduce 为 True，则会在 allreduce_op_handle 中调用 `cudaStreamSynchronize（nccl_stream）` ，这种模式在某些情况下可以获得更好的性能。

取值范围
---------------
Bool 型，缺省值为 True。

示例
-------
FLAGS_sync_nccl_allreduce=True - 在 allreduce_op_handle 中调用 `cudaStreamSynchronize(nccl_stream)` 。


FLAGS_tracer_profile_fname
*******************************************
(始于 1.4.0)

FLAGS_tracer_profile_fname 表示由 gperftools 生成的命令式跟踪器的分析器文件名。仅在编译选项选择`WITH_PROFILER = ON`时有效。如果禁用则设为 empty。

取值范围
---------------
String 型，缺省值为("gperf")。

示例
-------
FLAGS_tracer_profile_fname="gperf_profile_file" - 将命令式跟踪器的分析器文件名设为"gperf_profile_file"。
