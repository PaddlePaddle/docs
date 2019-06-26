
环境变量FLAGS
==================


allocator_strategy
********************
(始于1.2)

用于选择PaddlePaddle的分配器策略。 分配器策略正在开发中，且非legacy分配器尚未稳定。

取值范围
---------------
String型，['legacy', 'naive_best_fit']中的一个。缺省值为'legacy'。

示例
--------
FLAGS_allocator_strategy=legacy - 使用legacy分配器。

FLAGS_allocator_strategy=naive_best_fit - 使用新设计的分配器。


benchmark
********************
(始于0.12.0)

用于基准测试。设置后，它将使局域删除同步，添加一些内存使用日志，并在内核启动后同步所有cuda内核。

取值范围
---------------
Bool型，缺省值为False。

示例
-------
FLAGS_benchmark=True -  同步以测试基准。


check_nan_inf
********************
(始于0.13.0)

用于调试。它用于检查Operator的结果是否含有Nan或Inf。

取值范围
---------------
Bool型，缺省值为False。

示例
-------
FLAGS_check_nan_inf=True - 检查Operator的结果是否含有Nan或Inf。


communicator_fake_rpc
**********************
(始于1.5.0)

当设为True时，通信器不会实际进行rpc调用，因此速度不会受到网络通信的影响。该flag用于调试。

取值范围
---------------
Bool型，缺省值为False。

示例
-------
FLAGS_communicator_fake_rpc=True - 启用通信器fake模式。

注意
-------
该flag仅用于paddlepaddle的开发者，普通用户不应对其设置。


communicator_independent_recv_thread
**************************************
(始于1.5.0)

使用独立线程以从参数服务器接收参数。

取值范围
---------------
Bool型，缺省值为True。

示例
-------
FLAGS_communicator_independent_recv_thread=True - 使用独立线程以从参数服务器接收参数。

注意
-------
开发者使用该flag进行框架的调试与优化，普通用户不应对其设置。


communicator_max_merge_var_num
**************************************
(始于1.5.0)

要通过通信器合并为一个梯度并发送的最大梯度数。训练器将所有梯度放入队列，然后通信器将从队列中取出梯度并在合并后发送。

取值范围
---------------
Int32型，缺省值为20。

示例
-------
FLAGS_communicator_max_merge_var_num=16 - 将要通过通信器合并为一个梯度并发送的最大梯度数设为16。

注意
-------
该flag和训练器线程数有着密切关联，缺省值应和线程数一致。


communicator_merge_sparse_grad
*******************************************
(始于1.5.0)

在发送之前，合并稀疏梯度。

取值范围
---------------
Bool型，缺省值true。

示例
-------
FLAGS_communicator_merge_sparse_grad=true - 设置合并稀疏梯度。

注意
-------
合并稀疏梯度会耗费时间。如果重复ID较多，内存占用会变少，通信会变快；如果重复ID较少，则并不会节约内存。


communicator_min_send_grad_num_before_recv
*******************************************
(始于1.5.0)

在通信器中，有一个发送线程向参数服务器发送梯度，一个接收线程从参数服务器接收参数，且它们之间彼此独立。该flag用于控制接收线程的频率。 仅当发送线程至少发送communicator_min_send_grad_num_before_recv数量的梯度时，接收线程才会从参数服务器接收参数。

取值范围
---------------
Int32型，缺省值为20。

示例
-------
FLAGS_communicator_min_send_grad_num_before_recv=10 - 在接收线程从参数服务器接收参数之前，发送线程发送的梯度数为10。

注意
-------
由于该flag和训练器的训练线程数强相关，而每个训练线程都会发送其梯度，所以缺省值应和线程数一致。


communicator_send_queue_size
*******************************************
(始于1.5.0)

每个梯度的队列大小。训练器将梯度放入队列，然后通信器将其从队列中取出并发送出去。 当通信器很慢时，队列可能会满，训练器在队列有空间之前被持续阻塞。它用于避免训练比通信快得多，以致太多的梯度没有及时发出的情况。

取值范围
---------------
Int32型，缺省值为20。

示例
-------
FLAGS_communicator_send_queue_size=10 - 设置每个梯度的队列大小为10。

注意
-------
该flag会影响训练速度，若队列大小过大，速度会变快但结果可能会变差。


communicator_send_wait_times
*******************************************
(始于1.5.0)

合并数没有达到max_merge_var_num的情况下发送线程等待的次数。

取值范围
---------------
Int32型，缺省值为5。

示例
-------
FLAGS_communicator_send_wait_times=5 - 将合并数没有达到max_merge_var_num的情况下发送线程等待的次数设为5。


communicator_thread_pool_size
*******************************************
(始于1.5.0)

设置用于发送梯度和接收参数的线程池大小。

取值范围
---------------
Int32型，缺省值为5。

示例
-------
FLAGS_communicator_thread_pool_size=10 - 设置线程池大小为10。

注意
-------
大部分情况下，用户不需要设置该flag。


conv_workspace_size_limit
*******************************************
(始于0.13.0)

用于选择cuDNN卷积算法的工作区限制大小（单位为MB）。cuDNN的内部函数在这个内存限制范围内获得速度最快的匹配算法。通常，在较大的工作区内可以选择更快的算法，但同时也会显著增加内存空间。用户需要在内存和速度之间进行权衡。

取值范围
---------------
Uint64型，缺省值为4096。即4G内存工作区。

示例
-------
FLAGS_conv_workspace_size_limit=1024 - 将用于选择cuDNN卷积算法的工作区限制大小设置为1024MB。


cpu_deterministic
*******************************************
(始于0.15.0)

该flag用于调试。它表示是否在CPU侧确定计算结果。 在某些情况下，不同求和次序的结果可能不同，例如，`a+b+c+d` 的结果可能与 `c+a+b+d` 的结果不同。

取值范围
---------------
Bool型，缺省值为False。

示例
-------
FLAGS_cpu_deterministic=True - 在CPU侧确定计算结果。


cudnn_batchnorm_spatial_persistent
*******************************************
(始于1.4.0)

表示是否在batchnorm中使用新的批量标准化模式CUDNN_BATCHNORM_SPATIAL_PERSISTENT函数。

取值范围
---------------
Bool型，缺省值为False。

示例
-------
FLAGS_cudnn_batchnorm_spatial_persistent=True - 开启CUDNN_BATCHNORM_SPATIAL_PERSISTENT模式。

注意
-------
此模式在某些任务中可以更快，因为将为CUDNN_DATA_FLOAT和CUDNN_DATA_HALF数据类型选择优化路径。我们默认将其设置为False的原因是此模式可能使用原子整数缩减(scaled atomic integer reduction)而导致某些输入数据范围的数字溢出。


cudnn_deterministic
*******************************************
(始于0.13.0)

cuDNN对于同一操作有几种算法，一些算法结果是非确定性的，如卷积算法。该flag用于调试。它表示是否选择cuDNN中的确定性函数。 

取值范围
---------------
Bool型，缺省值为False。

示例
-------
FLAGS_cudnn_deterministic=True - 选择cuDNN中的确定性函数。

注意
-------
现在，在cuDNN卷积和池化Operator中启用此flag。确定性算法速度可能较慢，因此该flag通常用于调试。


cudnn_exhaustive_search
*******************************************
(始于1.2.0)

表示是否使用穷举搜索方法来选择卷积算法。在cuDNN中有两种搜索方法，启发式搜索和穷举搜索。穷举搜索尝试所有cuDNN算法以选择其中最快的算法。此方法非常耗时，所选择的算法将针对给定的层规格进行缓存。 一旦更改了图层规格（如batch大小，feature map大小），它将再次搜索。

取值范围
---------------
Bool型，缺省值为False。

示例
-------
FLAGS_cudnn_exhaustive_search=True - 使用穷举搜索方法来选择卷积算法。


dist_threadpool_size
*******************************************
(始于1.0.0)

控制用于分布式模块的线程数。如果未设置，则将其设置为硬线程。

取值范围
---------------
Int32型，缺省值为0。

示例
-------
FLAGS_dist_threadpool_size=10 - 将用于分布式模块的最大线程数设为10。


eager_delete_scope
*******************************************
(始于0.12.0)

同步局域删除。设置后，它将降低GPU内存使用量，但同时也会减慢销毁变量的速度（性能损害约1％）。

取值范围
---------------
Bool型，缺省值为True。

示例
-------
FLAGS_eager_delete_scope=True - 同步局域删除。


eager_delete_tensor_gb
*******************************************
(始于1.0.0)

表示是否使用垃圾回收策略来优化网络的内存使用。如果FLAGS_eager_delete_tensor_gb >= 0，则启用垃圾回收策略，并在运行网络时回收内存垃圾，这有利于节省内存使用量。它仅在您使用Executor运行程序、编译程序或使用并行数据编译程序时才有用。如果FLAGS_eager_delete_tensor_gb < 0，则禁用垃圾回收策略。垃圾回收器直到垃圾的内存大小达到FLAGS_eager_delete_tensor_gb GB时才会释放内存垃圾。

取值范围
---------------
Double型，单位为GB，缺省值为-1.0。

示例
-------
FLAGS_eager_delete_tensor_gb=0.0 - 一旦不再使用即释放内存垃圾。

FLAGS_eager_delete_tensor_gb=1.0 - 垃圾占用内存大小达到1.0GB时释放内存垃圾。

FLAGS_eager_delete_tensor_gb=-1.0 - 禁用垃圾回收策略。    

注意
-------
建议用户在训练大型网络时设置FLAGS_eager_delete_tensor_gb=0.0以启用垃圾回收策略。


enable_cublas_tensor_op_math
*******************************************
(始于1.2.0)

该flag表示是否使用Tensor Core，但可能会因此降低部分精确度。

取值范围
---------------
Bool型，缺省值为False。

示例
-------
enable_cublas_tensor_op_math=True - 使用Tensor Core。


enable_inplace_whitelist
*******************************************
(始于1.4)

该flag用于调试，在某些ops中禁止内存原位复用。设置后，一些ops不会执行原位复用优化以节省内存。这些Ops包括：sigmoid, exp, relu, tanh, sqrt, ceil, floor, reciprocal, relu6, soft_relu, hard_sigmoid, batch_norm, batch_norm_grad, sum, sum_grad, scale, reshape, elementwise_add, and elementwise_add_grad。

取值范围
---------------
Bool型，缺省值为False。

示例
-------
FLAGS_enable_inplace_whitelist=True - 在特定op上禁止内存原位复用优化。


enable_parallel_graph
*******************************************
(始于1.2.0)

该flag用于ParallelExecutor以禁用并行图执行模式。

取值范围
---------------
Bool型，缺省值为False。

示例
-------
FLAGS_enable_parallel_graph=False - 通过ParallelExecutor强制禁用并行图执行模式。


enable_rpc_profiler
*******************************************
(始于1.0.0)

是否启用RPC分析器。

取值范围
----------------
Bool型，缺省值为False。

示例
-------
FLAGS_enable_rpc_profiler=True - 启用RPC分析器并在分析器文件中记录时间线。


fast_eager_deletion_mode
*******************************************
(始于1.3)

是否使用快速垃圾回收策略。如果未设置，则在CUDA内核结束时释放gpu内存。否则gpu内存将在CUDA内核尚未结束的情况下被释放，从而使垃圾回收策略更快。仅在启用垃圾回收策略时有效。

取值范围
---------------
Bool型，缺省值为True。

示例
-------
FLAGS_fast_eager_deletion_mode=True - 启用快速垃圾回收策略。

FLAGS_fast_eager_deletion_mode=False - 禁用快速垃圾回收策略。


fraction_of_gpu_memory_to_use
*******************************************
(始于1.2.0)

表示分配的内存块占GPU总内存大小的比例。将来的内存使用将从该内存块分配。 如果内存块没有足够的gpu内存，将从gpu请求分配与内存块同样大小的新的内存块，直到gpu没有足够的内存为止。

取值范围
---------------
Uint64型，大于0，表示初始分配的内存块占GPU内存的比例。

示例
-------
FLAGS_fraction_of_gpu_memory_to_use=0.1 - 分配总GPU内存大小的10%作为初始GPU 内存块。

注意
-------
Windows系列平台会将FLAGS_fraction_of_gpu_memory_to_use默认设为0.5，Linux则会默认设为0.92。


free_idle_memory
*******************************************
(始于0.15.0)

是否在运行时释放从系统预分配的空闲内存。设置后，如果预分配的分配器中有太多空闲内存，则释放空闲内存。

取值范围
---------------
Bool型，缺省值为False。

示例
-------
FLAGS_free_idle_memory=True - 空闲内存太多时释放。

FLAGS_free_idle_memory=False - 不释放空闲内存。


fuse_parameter_groups_size
*******************************************
(始于1.4.0)

FLAGS_fuse_parameter_groups_size表示每一组中参数的个数。缺省值是一个经验性的结果。如果fuse_parameter_groups_size为1，则表示组的大小和参数梯度的数目一致。 如果fuse_parameter_groups_size为-1，则表示只有一个组。缺省值为3，这只是一个经验值。

取值范围
---------------
Int32型，缺省值为3。

示例
-------
FLAGS_fuse_parameter_groups_size=3 - 将单组参数的梯度大小设为3。


fuse_parameter_memory_size
*******************************************
(始于1.5.0)

FLAGS_fuse_parameter_memory_size表示作为通信调用输入（例如NCCLAllReduce）的单组参数梯度的上限内存大小。默认值为-1.0，表示不根据memory_size设置组。单位是MB。

取值范围
---------------
Double型，缺省值为-1.0。

示例
-------
FLAGS_fuse_parameter_memory_size=16 - 将单组参数梯度的上限大小设为16MB。


init_allocated_mem
*******************************************
(始于0.15.0)

是否对分配的内存进行非零值初始化。该flag用于调试，以防止某些Ops假定已分配的内存都是初始化为零的。

取值范围
---------------
Bool型，缺省值为False。

示例
-------
FLAGS_init_allocated_mem=True - 对分配的内存进行非零初始化。

FLAGS_init_allocated_mem=False - 不会对分配的内存进行非零初始化。


initial_cpu_memory_in_mb
*******************************************
(始于0.14.0)

初始PaddlePaddle分配器的CPU内存块大小，单位为MB。分配器将FLAGS_initial_cpu_memory_in_mb和FLAGS_fraction_of_cpu_memory_to_use*（总物理内存）的最小值作为内存块大小。

取值范围
---------------
Uint64型，缺省值为500，单位为MB。

示例
-------
FLAGS_initial_cpu_memory_in_mb=100 - 在FLAGS_fraction_of_cpu_memory_to_use*（总物理内存）大于100MB的情况下，首次提出分配请求时，分配器预先分配100MB内存，并在预分配的内存耗尽时再次分配100MB。


initial_gpu_memory_in_mb
*******************************************
(始于1.4.0)

分配一块指定大小的GPU内存块。之后的内存使用将从该内存块分配。如果内存块没有足够的gpu内存，将从gpu请求大小为FLAGS_reallocate_gpu_memory_in_mb的内存块，直到gpu没有剩余内存为止。

取值范围
---------------
Uint64型，大于0，为初始GPU内存大小，单位为MB。

示例
-------
FLAGS_initial_gpu_memory_in_mb=4096 - 分配4GB作为初始GPU内存块大小。

注意
-------
如果设置该flag，则FLAGS_fraction_of_gpu_memory_to_use设置的内存大小将被该flag覆盖。如果未设置该flag，PaddlePaddle将使用FLAGS_fraction_of_gpu_memory_to_use分配GPU内存。


inner_op_parallelism
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


limit_of_tmp_allocation
*******************************************
(始于1.3)

FLAGS_limit_of_tmp_allocation表示temporary_allocation大小的上限，单位为字节。如果FLAGS_limit_of_tmp_allocation为-1，temporary_allocation的大小将没有限制。

取值范围
---------------
Int64型，缺省值为-1。

示例
-------
FLAGS_limit_of_tmp_allocation=1024 - 将temporary_allocation大小的上限设为1024字节。


max_body_size
*******************************************
(始于1.0.0)

控制BRPC中的最大消息大小。

取值范围
---------------
Int32型，缺省值为2147483647。

示例
-------
FLAGS_max_body_size=2147483647 - 将BRPC消息大小设为2147483647。


memory_fraction_of_eager_deletion
*******************************************
(始于1.4)

垃圾回收策略释放变量的内存大小百分比。如果FLAGS_memory_fraction_of_eager_deletion = 1.0，则将释放网络中的所有临时变量。如果FLAGS_memory_fraction_of_eager_deletion = 0.0，则不会释放网络中的任何临时变量。如果0.0<FLAGS_memory_fraction_of_eager_deletion<1.0，则所有临时变量将根据其内存大小降序排序，并且仅
释放具有最大内存大小的FLAGS_memory_fraction_of_eager_deletion比例的变量。该flag仅在运行并行数据编译程序时有效。

取值范围
---------------
Double型，范围为[0.0, 1.0]，缺省值为1.0。

示例
-------
FLAGS_memory_fraction_of_eager_deletion=0 - 保留所有临时变量，也就是禁用垃圾回收策略。

FLAGS_memory_fraction_of_eager_deletion=1 - 释放所有临时变量。   

FLAGS_memory_fraction_of_eager_deletion=0.5 - 仅释放50%比例的占用内存最多的变量。


multiple_of_cupti_buffer_size
*******************************************
(始于1.4.0)

该flag用于分析。它表示CUPTI设备缓冲区大小的倍数。如果在profiler过程中程序挂掉或者在chrome://tracing中加载timeline文件时出现异常，请尝试增大此值。

取值范围
---------------
Int32型，缺省值为1。

示例
-------
FLAGS_multiple_of_cupti_buffer_size=1 - 将CUPTI设备缓冲区大小的倍数设为1。


paddle_num_threads
*******************************************
(始于0.15.0)

控制每个paddle实例的线程数。

取值范围
---------------
Int32型，缺省值为1。

示例
-------
FLAGS_paddle_num_threads=2 - 将每个实例的最大线程数设为2。


pe_profile_fname
*******************************************
(始于1.3.0)

该flag用于ParallelExecutor的调试。ParallelExecutor会通过gpertools生成配置文件结果，并将结果存储在FLAGS_pe_profile_fname指定的文件中。仅在编译选项选择 `WITH_PRIFILER=ON` 时有效。如果禁用则设为empty。

取值范围
---------------
String型，缺省值为empty ("")。

示例
-------
FLAGS_pe_profile_fname="./parallel_executor.perf" - 将配置文件结果存储在parallel_executor.perf中。


print_sub_graph_dir
*******************************************
(始于1.2.0)

该flag用于调试。如果程序中转换图的某些子图失去连接，则结果可能会出错。我们可以将这些断开连接的子图打印到该flag指定的文件中。如果禁用则设为empty。

取值范围
---------------
String型，缺省值为empty ("")。

示例
-------
FLAGS_print_sub_graph_dir="./sub_graphs.txt" - 将断开连接的子图打印到"./sub_graphs.txt"。


reader_queue_speed_test_mode
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


reallocate_gpu_memory_in_mb
*******************************************
(始于1.4.0)

如果耗尽了分配的GPU内存块，则重新分配额外的GPU内存块。

取值范围
---------------
Int64型，大于0，单位为MB。

示例
-------
FLAGS_reallocate_gpu_memory_in_mb=1024 - 如果耗尽了分配的GPU内存块，重新分配1GB。

注意
-------
如果设置了该flag，PaddlePaddle将重新分配该flag指定大小的gpu内存。否则分配FLAGS_fraction_of_gpu_memory_to_use指定比例的gpu内存。


rpc_deadline
*******************************************
(始于1.0.0)

它控制rpc通信的deadline超时。

取值范围
---------------
Int32型，缺省值为180000，单位为ms。

示例
-------
FLAGS_rpc_deadline=180000 - 将deadline超时设为3分钟。


rpc_disable_reuse_port
*******************************************
(始于1.2.0)

rpc_disable_reuse_port为True时，grpc的 GRPC_ARG_ALLOW_REUSEPORT会被设置为False以禁用SO_REUSEPORT。

取值范围
---------------
Bool型，缺省值为False。

示例
-------
FLAGS_rpc_disable_reuse_port=True - 禁用SO_REUSEPORT。


rpc_get_thread_num
*******************************************
(始于1.0.0)

它控制用于从参数服务器获取参数的线程数。

取值范围
---------------
Int32型，缺省值为12。

示例
-------
FLAGS_rpc_get_thread_num=6 - 将从参数服务器获取参数的线程数设为6。


rpc_send_thread_num
*******************************************
(始于1.0.0)

它控制用于发送rpc的线程数。

取值范围
---------------
Int32型，缺省值为12。

示例
-------
FLAGS_rpc_send_thread_num=6 - 将用于发送的线程数设为6。


rpc_server_profile_path
*******************************************
since(v0.15.0)

设置分析器输出日志文件路径前缀。完整路径为rpc_server_profile_path_listener_id，其中listener_id为随机数。 

取值范围
---------------
String型，缺省值为"./profile_ps"。

示例
-------
FLAGS_rpc_server_profile_path="/tmp/pserver_profile_log" - 在"/tmp/pserver_profile_log_listener_id"中生成配置日志文件。


selected_gpus
*******************************************
(始于1.3)

设置用于训练或预测的GPU设备。

取值范围
---------------
以逗号分隔的设备ID列表，其中每个设备ID是一个非负整数，且应小于您的机器拥有的GPU设备总数。

示例
-------
FLAGS_selected_gpus=0,1,2,3,4,5,6,7 - 令0-7号GPU设备用于训练和预测。

注意
-------
使用该flag的原因是我们希望在GPU设备之间使用聚合通信，但通过CUDA_VISIBLE_DEVICES只能使用共享内存。


sync_nccl_allreduce
*******************************************
(始于1.3)

如果FLAGS_sync_nccl_allreduce为True，则会在allreduce_op_handle中调用 `cudaStreamSynchronize（nccl_stream）` ，这种模式在某些情况下可以获得更好的性能。

取值范围
---------------
Bool型，缺省值为True。

示例
-------
FLAGS_sync_nccl_allreduce=True - 在allreduce_op_handle中调用 `cudaStreamSynchronize(nccl_stream)` 。


times_excess_than_required_tmp_allocation
*******************************************
(始于1.3)

FLAGS_times_excess_than_required_tmp_allocation表示TemporaryAllocator可以返回的最大大小。例如，如果所需的内存大小为N，且times_excess_than_required_tmp_allocation为2.0，则TemporaryAllocator将返回大小范围为N~2*N的可用分配。

取值范围
---------------
Int64型，缺省值为2。

示例
-------
FLAGS_times_excess_than_required_tmp_allocation=1024 - 设置TemporaryAllocator可以返回的最大大小为1024*N。


tracer_profile_fname
*******************************************
(始于1.4.0)

FLAGS_tracer_profile_fname表示由gperftools生成的命令式跟踪器的分析器文件名。仅在编译选项选择`WITH_PROFILER = ON`时有效。如果禁用则设为empty。

取值范围
---------------
String型，缺省值为("gperf")。

示例
-------
FLAGS_tracer_profile_fname="gperf_profile_file" - 将命令式跟踪器的分析器文件名设为"gperf_profile_file"。


use_mkldnn
*******************************************
(始于0.13.0)

在预测或训练过程中，可以通过该选项选择使用Intel MKL-DNN（https://github.com/intel/mkl-dnn）库运行。
“用于深度神经网络的英特尔（R）数学核心库（Intel(R) MKL-DNN）”是一个用于深度学习应用程序的开源性能库。该库加速了英特尔（R）架构上的深度学习应用程序和框架。Intel MKL-DNN包含矢量化和线程化构建建块，您可以使用它们来实现具有C和C ++接口的深度神经网络（DNN）。

取值范围
---------------
Bool型，缺省值为False。

示例
-------
FLAGS_use_mkldnn=True - 开启使用MKL-DNN运行。

注意
-------
FLAGS_use_mkldnn仅用于python训练和预测脚本。要在CAPI中启用MKL-DNN，请设置选项 -DWITH_MKLDNN=ON。
英特尔MKL-DNN支持英特尔64架构和兼容架构。
该库对基于以下设备的系统进行了优化：
英特尔SSE4.1支持的英特尔凌动（R）处理器；
第4代，第5代，第6代，第7代和第8代英特尔（R）Core（TM）处理器；
英特尔（R）Xeon（R）处理器E3，E5和E7系列（原Sandy Bridge，Ivy Bridge，Haswell和Broadwell）；
英特尔（R）Xeon（R）可扩展处理器（原Skylake和Cascade Lake）；
英特尔（R）Xeon Phi（TM）处理器（原Knights Landing and Knights Mill）；
兼容处理器。


use_ngraph
*******************************************
(始于1.4.0)

在预测或训练过程中，可以通过该选项选择使用英特尔nGraph（https://github.com/NervanaSystems/ngraph）引擎。它将在英特尔Xeon CPU上获得很大的性能提升。

取值范围
---------------
Bool型，缺省值为False。

示例
-------
FLAGS_use_ngraph=True - 开启使用nGraph运行。

注意
-------
英特尔nGraph目前仅在少数模型中支持。我们只验证了[ResNet-50]（https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/README_ngraph.md）的训练和预测。


use_pinned_memory
*******************************************
(始于0.12.0)

是否使用pinned memory。设为True后，CPU分配器将调用mlock来锁定内存页。

取值范围
---------------
Bool型，缺省值为True。

示例
-------
FLAGS_use_pinned_memory=True - 锁定分配的CPU内存页面。
