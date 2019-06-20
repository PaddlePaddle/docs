==================
FLAGS
==================

allocator_strategy
**************************************
(since 1.2)

Use to choose allocator strategy of PaddlePaddle. The allocator strategy is under development, and the non-legacy allocator is not stable yet.

Values accepted
---------------
String, enum in ['legacy', 'naive_best_fit']. The default value is 'legacy'.

Example
--------
FLAGS_allocator_strategy=legacy would use the legacy allocator.

FLAGS_allocator_strategy=naive_best_fit would use the new-designed allocator.


benchmark
**************************************
(since 0.12.0)

Used to do benchmark. If set, it will make scope delete synchronized, add some memory usage log, and synchronize all cuda kernel after kernel launches.

Values accepted
---------------
Bool. The default value is False.

Example
-------
FLAGS_benchmark=True will do some synchronizations to test benchmark.


check_nan_inf
**************************************
(since 0.13.0)

This Flag is used for debugging. It is used to check whether the result of the Operator has Nan or Inf.

Values accepted
---------------
Bool. The default value is False.

Example
-------
FLAGS_check_nan_inf=True will check the result of Operator whether the result has Nan or Inf.


communicator_fake_rpc
**************************************
(since 1.5.0)

When set true, communicator will not really do rpc call, so the speed will not be affected by network communication. This flag is used for debugging purpose.

Values accepted
---------------
Bool. The default value is false.

Example
-------
FLAGS_communicator_fake_rpc=True will enable communicator fake mode.

Note
-------
This flag is only for developer of paddlepaddle, user should not set it.


communicator_independent_recv_thread
**************************************
(since 1.5.0)

use an independent thread to receive parameter from parameter server

Values accepted
---------------
Bool. The default value is True.

Example
-------
FLAGS_communicator_independent_recv_thread=True will use an independent thread to receive parameter from parameter server.

Note
-------
This flag is for developer to debug and optimize the framework. User should not set it.


communicator_max_merge_var_num
**************************************
(since 1.5.0)

max gradient number to merge and send as one gradient by communicator. Trainer will put all gradients into a queue, then communicator will take the gradients out from the queue and merge them before send.

Values accepted
---------------
Int32. The default value is 20.

Example
-------
FLAGS_communicator_max_merge_var_num=16 will set the max gradient number to merge and send as one gradient to 16.

Note
-------
This flag has strong relationship with trainer thread num. The default value should be the same with thread num.


communicator_min_send_grad_num_before_recv
*******************************************
(since 1.5.0)

In communicator, there is one send thread that send gradient to parameter server and one receive thread that receive parameter from parameter server. They work independently. This flag is used to control the frequency of receive thread. Only when the send thread send at least communicator_min_send_grad_num_before_recv gradients will the receive thread receive parameter from parameter server.

Values accepted
---------------
Int32. The default value is 20.

Example
-------
FLAGS_communicator_min_send_grad_num_before_recv=10 will set the number of gradients sent by the send thread to 10 before the receive thread receive parameter from parameter server.

Note
-------
This flag has strong relation with the training threads of trainer. because each training thread will send it's grad. So the default value should be training thread num.


communicator_send_queue_size
*******************************************
(since 1.5.0)

The queue size for each gradient. Trainer will put gradient into a queue, and communicator will take gradient out from the queue and then send them out. When communicator is slow, the queue may be full and then the trainer will be blocked until the queue has space. It's used to avoid the situation that training is much more faster than communication. There will be too much gradients that is not sent out in time.

Values accepted
---------------
Int32. The default value is 20.

Example
-------
FLAGS_communicator_send_queue_size=10 will set the queue size for each gradient to 10.

Note
-------
This flag will affect the training speed, if the queue size is larger, the speed may be faster, but may make the result worse.


communicator_send_wait_times
*******************************************
(since 1.5.0)

times that send thread will wait if merge number does not reach max_merge_var_num.

Values accepted
---------------
Int32. The default value is 5.

Example
-------
FLAGS_communicator_send_wait_times=5 set the times that send thread will wait if merge number does not reach max_merge_var_num to 5.


communicator_thread_pool_size
*******************************************
(since 1.5.0)

Set the thread pool size that used to do gradient send and parameter receive.

Values accepted
---------------
Int32. The default value is 5.

Example
-------
FLAGS_communicator_thread_pool_size=10 set the thread pool size to 10.

Note
-------
Most of time user does not need to set this flag.


conv_workspace_size_limit
*******************************************
(since 0.13.0)

The workspace limit size in MB unit for choosing cuDNN convolution algorithms. The inner funciton of cuDNN obtain the fastest suited algorithm that fits within this memory limit. Usually, large workspace size may lead to choose faster algorithms, but significant increasing memory workspace. Users need to trade-off between memory and speed.

Values accepted
---------------
Uint64. The default value is 4096. That is to say, 4G memory workspace.

Example
-------
FLAGS_conv_workspace_size_limit=1024 set the workspace limit size for choosing cuDNN convolution algorithms to 1024MB.


cpu_deterministic
*******************************************
(since 0.15.0)

This Flag is used for debugging. It indicates whether to make the result of computation deterministic in CPU side. In some case, the result of the different order of summing maybe different，for example, the result of `a+b+c+d` may be different with the result of `c+a+b+d`.

Values accepted
---------------
Bool. The default value is False.

Example
-------
FLAGS_cpu_deterministic=True will make the result of computation deterministic in CPU side.


cudnn_batchnorm_spatial_persistent
*******************************************
(since 1.4.0)

Indicates whether to use the new batch normalization mode CUDNN_BATCHNORM_SPATIAL_PERSISTENT function in batchnorm.

Values accepted
---------------
Bool. The default value is False.

Example
-------
FLAGS_cudnn_batchnorm_spatial_persistent=True will enable the CUDNN_BATCHNORM_SPATIAL_PERSISTENT mode.

Note
-------
This mode can be faster in some tasks because an optimized path will be selected for CUDNN_DATA_FLOAT and CUDNN_DATA_HALF data types. The reason we set it to False by default is that this mode may use scaled atomic integer reduction which may cause a numerical overflow for some input data range.


cudnn_deterministic
*******************************************
(since 0.13.0)

For one operation, cuDNN has several algorithms, some algorithm results are non-deterministic, like convolution algorithms. This flag is used for debugging. It indicates whether to choose the deterministic in cuDNN. 

Values accepted
---------------
Bool. The default value is False.

Example
-------
FLAGS_cudnn_deterministic=True will choose the deterministic in cuDNN.

Note
-------
Now this flag is enabled in cuDNN convolution and pooling operator. The deterministic algorithms may slower, so this flag is generally used for debugging.


cudnn_exhaustive_search
*******************************************
(since 1.2.0)

Whether to use exhaustive search method to choose convolution algorithms. There are two search methods, heuristic search and exhaustive search in cuDNN. The exhaustive search attempts all cuDNN algorithms to choose the fastest algorithm. This method is time-consuming, the choosed algorithm will be cached for the given layer specifications. Once the layer specifications (like batch size, feature map size) are changed, it will search again.

Values accepted
---------------
Bool. The default value is False. 

Example
-------
FLAGS_cudnn_exhaustive_search=True will use exhaustive search method to choose convolution algorithms.


dist_threadpool_size
*******************************************
(Since 1.0.0)

Control the number of thread used for distributed module. If it's not set, it will be set to hardware threads.

Values accepted
---------------
Int32. The default value is 0.

Example
-------
FLAGS_dist_threadpool_size=10 will enable 10 threads as max number of thread used for distributed module.


eager_delete_scope
*******************************************
(since 0.12.0)

Make scope delete synchronously. If set, it will reduce GPU memory usage but slow down the destruction of variables (around 1% performance harm).

Values accepted
---------------
Bool. The default value is True.

Example
-------
FLAGS_eager_delete_scope=True will make scope delete synchronously.


eager_delete_tensor_gb
*******************************************
(since 1.0.0)

Whether to use garbage collection strategy to optimize the memory usage of network. If FLAGS_eager_delete_tensor_gb >= 0, garbage collection strategy would be enabled, and collect memory garbages when running network, which is beneficial to saving memory usage. It is only useful when you use Executor to run program, or compile program, or compile program with data parallel. If FLAGS_eager_delete_tensor_gb < 0, garbage collection strategy is disabled. Garbage collector would not release memory garbages until the memory size of garbages reaches FLAGS_eager_delete_tensor_gb GB.

Values accepted
---------------
Double, in GB unit. The default value is -1.0.

Example
-------
FLAGS_eager_delete_tensor_gb=0.0 would make memory garbage release immediately once it is not used. 

FLAGS_eager_delete_tensor_gb=1.0 would make memory garbage release till the memory size of garbages reaches 1.0GB. 

FLAGS_eager_delete_tensor_gb=-1.0 would disable garbage collection strategy.

Note
-------
It is recommended that users enable garbage collection strategy by setting FLAGS_eager_delete_tensor_gb=0.0 when training large network.


enable_cublas_tensor_op_math
*******************************************
(since 1.2.0)

This Flag indicates whether to use Tensor Core, but it may lose some precision. 

Values accepted
---------------
Bool. The default value is False.

Example
-------
enable_cublas_tensor_op_math=True will use Tensor Core.


enable_inplace_whitelist
*******************************************
(since 1.4)

Debug use to disable memory in-place in some ops. If set, some ops would not perform in-place optimization to save memory. These ops include: sigmoid, exp, relu, tanh, sqrt, ceil, floor, reciprocal, relu6, soft_relu, hard_sigmoid, batch_norm, batch_norm_grad, sum, sum_grad, scale, reshape, elementwise_add, and elementwise_add_grad.

Values accepted
---------------
Bool. The default value is False.

Example
-------
FLAGS_enable_inplace_whitelist=True would disable memory in-place optimization on certain ops.


enable_parallel_graph
*******************************************
(since 1.2.0)

This Flag is used for ParallelExecutor to disable parallel graph execution mode.

Values accepted
---------------
Bool. The default value is False.

Example
-------
FLAGS_enable_parallel_graph=False will force disable parallel graph execution mode by ParallelExecutor.


enable_rpc_profiler
*******************************************
(Since 1.0.0)

Enable RPC profiler or not.

Values accepted
----------------
Bool. The default value is False.

Example
-------
FLAGS_enable_rpc_profiler=True will enable rpc profiler and record the timeline to profiler file.


fast_eager_deletion_mode
*******************************************
(since 1.3)

Whether to use fast garbage collection strategy. If not set, gpu memory would be released when CUDA kernel ends. Otherwise, gpu memory would be released without waiting CUDA kernel ends, making garbage collection strategy faster. Only valid when garbage collection strategy is enabled.

Values accepted
---------------
Bool. The default value is True.

Example
-------
FLAGS_fast_eager_deletion_mode=True would turn on fast garbage collection strategy. 

FLAGS_fast_eager_deletion_mode=False would turn off fast garbage collection strategy.


fraction_of_gpu_memory_to_use
*******************************************
(since 1.2.0)

Allocate a chunk of gpu memory that is this fraction of the total gpu memory size. Future memory usage will be allocated from the chunk. If the chunk doesn't have enough gpu memory, additional chunks of the same size will be requested from gpu until the gpu has no memory left for another chunk.

Values accepted
---------------
Uint64 value greater than 0 which is the initial GPU memory percentage.

Example
-------
FLAGS_fraction_of_gpu_memory_to_use=0.1 will allocate 10% total gpu memory size as initial GPU chunk.

Note
-------
Windows series platform will set FLAGS_fraction_of_gpu_memory_to_use to 0.5 by default.
Linux will set FLAGS_fraction_of_gpu_memory_to_use to 0.92 by default.


free_idle_memory
*******************************************
(since 0.15.0)

Whether to free idle memory pre-allocated from system during runtime. If set, free idle memory would be released if there is too much free idle memory in the pre-allocated allocator.

Values accepted
---------------
Bool. The default value is False.

Example
-------
FLAGS_free_idle_memory=True will free idle memory when there is too much of it. 

FLAGS_free_idle_memory=False will not free idle memory.


fuse_parameter_groups_size
*******************************************
(since 1.4.0)

FLAGS_fuse_parameter_groups_size is the size of one group parameters' gradient. The default value is an empirical result. If the fuse_parameter_groups_size is 1, it means that the groups' size is the number of parameters' gradient. If the fuse_parameter_groups_size is -1, it means that there is only one group. The default value is 3, it is an empirical value.

Values accepted
---------------
Int32. The default value is 3.

Example
-------
FLAGS_fuse_parameter_groups_size=3 will set the size of one group parameters' gradient to 3.


fuse_parameter_memory_size
*******************************************
(since 1.5.0)

FLAGS_fuse_parameter_memory_size indicates the up limited memory size of one group parameters' gradient which is the input of communication calling ( e.g NCCLAllReduce). The default value is -1.0, it means that not set group according to memory_size. The unit is Megabyte.

Values accepted
---------------
Double. The default value is -1.0.

Example
-------
FLAGS_fuse_parameter_memory_size=16 set the up limited memory size of one group parameters' gradient to 16 Megabytes.


init_allocated_mem
*******************************************
(since 0.15.0)

Whether to initialize the allocated memory by some non-zero values. This flag is for debug use to prevent that some ops assumes that the memory allocated is initialized to be zero.

Values accepted
---------------
Bool. The default value is False.

Example
-------
FLAGS_init_allocated_mem=True will make the allocated memory initialize as a non-zero value. 

FLAGS_init_allocated_mem=False will not initialize the allocated memory.


initial_cpu_memory_in_mb
*******************************************
(since 0.14.0)

Initial CPU memory chunk size in MB of PaddlePaddle allocator. Allocator would take the minimal value of FLAGS_initial_cpu_memory_in_mb and FLAGS_fraction_of_cpu_memory_to_use*(total physical memory) as the memory chunk size.

Values accepted
---------------
Uint64. The default value is 500 with unit MB.

Example
-------
FLAGS_initial_cpu_memory_in_mb=100, if FLAGS_fraction_of_cpu_memory_to_use*(total physical memory) > 100MB, then allocator will pre-allocate 100MB when first allocation request raises, and re-allocate 100MB again when the pre-allocated memory is exhaustive.


initial_gpu_memory_in_mb
*******************************************
(since 1.4.0)

Allocate a chunk of GPU memory whose byte size is specified by the flag. Future memory usage will be allocated from the chunk. If the chunk doesn't have enough gpu memory, additional chunks of the gpu memory will be requested from gpu with size specified by FLAGS_reallocate_gpu_memory_in_mb until the gpu has no memory left for the additional chunk.

Values accepted
---------------
Uint64 value greater than 0 which is the initial GPU memory size in MB.

Example
-------
FLAGS_initial_gpu_memory_in_mb=4096 will allocate 4 GB as initial GPU chunk.

Note
-------
If you set this flag, the memory size set by FLAGS_fraction_of_gpu_memory_to_use will be overrided by this flag.
If you don't set this flag, PaddlePaddle will use FLAGS_fraction_of_gpu_memory_to_use to allocate gpu memory.


inner_op_parallelism
*******************************************
(since 1.3.0)

Most operators are working in single thread mode, but for some operator, use multi thread is more suitable. For Example, optimization op that optimize sparse gradient will be much faster to use multi thread. This flag is used to set the thread number inside an operator.

Values accepted
---------------
Int32. The default value is 0 which means that operator will not run in multi thread mode.

Example
-------
FLAGS_inner_op_parallelism=5 will set the thread number inside an operator to 5.

Note
-------
currently only sparse adam op supports inner_op_parallelism.


limit_of_tmp_allocation
*******************************************
(since 1.3)

The FLAGS_limit_of_tmp_allocation indicates the up limit of temporary_allocation size, the unit is byte. If the FLAGS_limit_of_tmp_allocation is -1, the size of temporary_allocation will not be limited.

Values accepted
---------------
Int64. The default value is -1.

Example
-------
FLAGS_limit_of_tmp_allocation=1024 will set the up limit of temporary_allocation size to 1024 bytes.


max_body_size
*******************************************
(Since 1.0.0)

It controls the max message size in BRPC.

Values accepted
---------------
Int32. The default value is 2147483647.

Example
-------
FLAGS_max_body_size=2147483647 will set the BRPC message size to 2147483647.


memory_fraction_of_eager_deletion
*******************************************
(since 1.4)

A memory size percentage when garbage collection strategy decides which variables should be released. If FLAGS_memory_fraction_of_eager_deletion=1.0, all temporary variables in the network would be released. If FLAGS_memory_fraction_of_eager_deletion=0.0, all temporary variables in the network would not be released. If 0.0<FLAGS_memory_fraction_of_eager_deletion<1.0, all temporary variables would be sorted descendingly according to their memory size, and only 
FLAGS_memory_fraction_of_eager_deletion of variables with largest memory size would be released. This flag is only valid when running compiled program with data parallel.

Values accepted
---------------
Double, inside [0.0, 1.0]. The default value is 1.0.

Example
-------
FLAGS_memory_fraction_of_eager_deletion=0 would keep all temporary variables, that is to say, disabling garbage collection strategy.

FLAGS_memory_fraction_of_eager_deletion=1 would release all temporary variables.  
  
FLAGS_memory_fraction_of_eager_deletion=0.5 would only release 50% of variables with largest memory size.


multiple_of_cupti_buffer_size
*******************************************
(since 1.4.0)

This Flag is used for profiling. It indicates the multiple of the CUPTI device buffer size. When you are profiling, if the program breaks down or bugs rise when loading timeline file in chrome://traxing, try increasing this value.

Values accepted
---------------
Int32. The default value is 1.

Example
-------
FLAGS_multiple_of_cupti_buffer_size=1 set the multiple of the CUPTI device buffer size to 1.


paddle_num_threads
*******************************************
(since 0.15.0)

Control the number of threads of each paddle instance.

Values accepted
---------------
Int32. The default value is 1.

Example
-------
FLAGS_paddle_num_threads=2 will enable 2 threads as max number of threads for each instance.


pe_profile_fname
*******************************************
(since 1.3.0)

This Flag is used for debugging for ParallelExecutor. The ParallelExecutor will generate the profile result by gperftools, and the profile result will be stored in the file which is specified by FLAGS_pe_profile_fname. Only valid when compiled `WITH_PRIFILER=ON`. Empty if disable.

Values accepted
---------------
String. The default value is empty ("").

Example
-------
FLAGS_pe_profile_fname="./parallel_executor.perf" will store the profile result to parallel_executor.perf.


print_sub_graph_dir
*******************************************
(since 1.2.0)

This Flag is used for debugging. If some subgraphs of the transformed graph from the program are disconnected, the result may be problematic. We can print these disconnected subgraphs to a file specified by the flag. Empty if disable.

Values accepted
---------------
String. The default value is empty ("").

Example
-------
FLAGS_print_sub_graph_dir="./sub_graphs.txt" will print the disconnected subgraphs to "./sub_graphs.txt".


reader_queue_speed_test_mode
*******************************************
(since 1.1.0)

Set the pyreader data queue to test mode. In test mode, pyreader will cache some data, executor will then read the cached data, so reader will not be the bottleneck.

Values accepted
---------------
Bool. The default value is False.

Example
-------
FLAGS_reader_queue_speed_test_mode=True will enable the pyreader test mode.

Note
-------
This flag will work only when you are using py_reader.


reallocate_gpu_memory_in_mb
*******************************************
(since 1.4.0)

Re-allocate additional GPU chunk if run out of allocated GPU memory chunk.

Values accepted
---------------
Int64 value greater than 0 in MB

Example
-------
FLAGS_reallocate_gpu_memory_in_mb=1024 will re-allocate 1 GB if run out of GPU memory chunk.

Note
-------
If this flag is set, PaddlePaddle will reallocate the gpu memory with size specified by this flag.
Else PaddlePaddle will reallocate with size set by FLAGS_fraction_of_gpu_memory_to_use.


rpc_deadline
*******************************************
(Since 1.0.0)

It controls the deadline timeout of the rpc communication.

Values accepted
---------------
Int32. The default value is 180000 in ms.

Example
-------
FLAGS_rpc_deadline=180000 will set deadline timeout to 3 minute.


rpc_disable_reuse_port
*******************************************
(since 1.2.0)

When rpc_disable_reuse_port is true, the flag of grpc GRPC_ARG_ALLOW_REUSEPORT will be set to false to
disable the use of SO_REUSEPORT if it's available.

Values accepted
---------------
Bool. The default value is False.

Example
-------
FLAGS_rpc_disable_reuse_port=True will disable the use of SO_REUSEPORT.


rpc_get_thread_num
*******************************************
(Since 1.0.0)

It controls the number of threads used to get parameter from parameter server.

Values accepted
---------------
Int32. The default value is 12.

Example
-------
FLAGS_rpc_get_thread_num=6 will use 6 threads to get parameter from parameter server.


rpc_send_thread_num
*******************************************
(Since 1.0.0)

It controls the number of threads used for send rpc.

Values accepted
---------------
Int32. The default value is 12.

Example
-------
FLAGS_rpc_send_thread_num=6 will set number thread used for send to 6.


rpc_server_profile_path
*******************************************
since(v0.15.0)

Set the profiler output log file path prefix. The complete path will be rpc_server_profile_path_listener_id, listener_id is a random number.

Values accepted
---------------
String. The default value is "./profile_ps".

Example
-------
FLAGS_rpc_server_profile_path="/tmp/pserver_profile_log" generate profile log file at "/tmp/pserver_profile_log_listener_id".


selected_gpus
*******************************************
(since 1.3)

Set the GPU devices used for training or inference.

Values accepted
---------------
A comma-separated list of device IDs, where each device ID is a nonnegative integer less than the number of GPU devices your machine have.

Example
-------
FLAGS_selected_gpus=0,1,2,3,4,5,6,7 makes GPU devices 0-7 to be used for training or inference.

Note
-------
The reason for using this flag is that we want to use collective communication between GPU devices, but with CUDA_VISIBLE_DEVICES can only use share-memory.


sync_nccl_allreduce
*******************************************
(since 1.3)

If the FLAGS_sync_nccl_allreduce is true, there will call `cudaStreamSynchronize(nccl_stream)` in allreduce_op_handle, this mode can get better performance in some scenarios.

Values accepted
---------------
Bool. The default value is True.

Example
-------
FLAGS_sync_nccl_allreduce=True will call `cudaStreamSynchronize(nccl_stream)` in allreduce_op_handle.


times_excess_than_required_tmp_allocation
*******************************************
(since 1.3)

The FLAGS_times_excess_than_required_tmp_allocation indicates the max size the TemporaryAllocator can return. For Example
, if the required memory size is N, and times_excess_than_required_tmp_allocation is 2.0, the TemporaryAllocator will return the available allocation that the range of size is N ~ 2*N.

Values accepted
---------------
Int64. The default value is 2.

Example
-------
FLAGS_times_excess_than_required_tmp_allocation=1024 will set the max size of the TemporaryAllocator can return to 1024*N.


tracer_profile_fname
*******************************************
(since 1.4.0)

FLAGS_tracer_profile_fname indicates the profiler filename for imperative tracer, which generated by gperftools. Only valid when compiled `WITH_PROFILER=ON`. Empty if disabled.

Values accepted
---------------
String. The default value is ("gperf").

Example
-------
FLAGS_tracer_profile_fname="gperf_profile_file" will set the profiler filename for imperative tracer to "gperf_profile_file".


use_mkldnn
*******************************************
(since 0.13.0)

Give a choice to run with Intel MKL-DNN (https://github.com/intel/mkl-dnn) library on inference or training.

Intel(R) Math Kernel Library for Deep Neural Networks (Intel(R) MKL-DNN) is an open-source performance library for deep-learning applications. The library accelerates deep-learning applications and frameworks on Intel(R) architecture. Intel MKL-DNN contains vectorized and threaded building blocks that you can use to implement deep neural networks (DNN) with C and C++ interfaces.

Values accepted
---------------
Bool. The default value is False.

Example
-------
FLAGS_use_mkldnn=True will enable running with MKL-DNN support.

Note
-------
FLAGS_use_mkldnn is only used for python training and inference scripts. To enable MKL-DNN in CAPI, set build option -DWITH_MKLDNN=ON
Intel MKL-DNN supports Intel 64 architecture and compatible architectures. The library is optimized for the systems based on:
Intel Atom(R) processor with Intel SSE4.1 support
4th, 5th, 6th, 7th, and 8th generation Intel(R) Core(TM) processor
Intel(R) Xeon(R) processor E3, E5, and E7 family (formerly Sandy Bridge, Ivy Bridge, Haswell, and Broadwell)
Intel(R) Xeon(R) Scalable processors (formerly Skylake and Cascade Lake)
Intel(R) Xeon Phi(TM) processors (formerly Knights Landing and Knights Mill)
and compatible processors.


use_ngraph
*******************************************
(since 1.4.0)

Give a choice to run with Intel nGraph(https://github.com/NervanaSystems/ngraph) engine on inference or training. This will obtain much performance boost on Intel Xeon CPU.

Values accepted
---------------
Bool. The default value is False.

Example
-------
FLAGS_use_ngraph=True will enable running with nGraph support.

Note
-------
Intel nGraph is only supported in few models yet. We have only verified [ResNet-50](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/README_ngraph.md) training and inference.


use_pinned_memory
*******************************************
(since 0.12.0)

Whether to use cpu pinned memory. If set, CPU allocator calls mlock to lock pages.

Values accepted
---------------
Bool. The default value is True.

Example
-------
FLAGS_use_pinned_memory=True would make the pages of allocated cpu memory lock.
