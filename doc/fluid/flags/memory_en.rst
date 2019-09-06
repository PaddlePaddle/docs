
memory management
==================


FLAGS_allocator_strategy
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



FLAGS_eager_delete_scope
*******************************************
(since 0.12.0)

Make scope delete synchronously. If set, it will reduce GPU memory usage but slow down the destruction of variables (around 1% performance harm).

Values accepted
---------------
Bool. The default value is True.

Example
-------
FLAGS_eager_delete_scope=True will make scope delete synchronously.


FLAGS_eager_delete_tensor_gb
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



FLAGS_enable_inplace_whitelist
*******************************************
(since 1.4)

Debug use to disable memory in-place in some ops. If set, some ops would not perform in-place optimization to save memory. These ops include: sigmoid, exp, relu, tanh, sqrt, ceil, floor, reciprocal, relu6, soft_relu, hard_sigmoid, batch_norm, batch_norm_grad, sum, sum_grad, scale, reshape, elementwise_add, and elementwise_add_grad.

Values accepted
---------------
Bool. The default value is False.

Example
-------
FLAGS_enable_inplace_whitelist=True would disable memory in-place optimization on certain ops.



FLAGS_fast_eager_deletion_mode
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


FLAGS_fraction_of_gpu_memory_to_use
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


FLAGS_free_idle_memory
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


FLAGS_fuse_parameter_groups_size
*******************************************
(since 1.4.0)

FLAGS_fuse_parameter_groups_size is the size of one group parameters' gradient. The default value is an empirical result. If the fuse_parameter_groups_size is 1, it means that the groups' size is the number of parameters' gradient. If the fuse_parameter_groups_size is -1, it means that there is only one group. The default value is 3, it is an empirical value.

Values accepted
---------------
Int32. The default value is 3.

Example
-------
FLAGS_fuse_parameter_groups_size=3 will set the size of one group parameters' gradient to 3.



FLAGS_fuse_parameter_memory_size
*******************************************
(since 1.5.0)

FLAGS_fuse_parameter_memory_size indicates the up limited memory size of one group parameters' gradient which is the input of communication calling ( e.g NCCLAllReduce). The default value is -1.0, it means that not set group according to memory_size. The unit is Megabyte.

Values accepted
---------------
Double. The default value is -1.0.

Example
-------
FLAGS_fuse_parameter_memory_size=16 set the up limited memory size of one group parameters' gradient to 16 Megabytes.


FLAGS_init_allocated_mem
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


FLAGS_initial_cpu_memory_in_mb
*******************************************
(since 0.14.0)

Initial CPU memory chunk size in MB of PaddlePaddle allocator. Allocator would take the minimal value of FLAGS_initial_cpu_memory_in_mb and FLAGS_fraction_of_cpu_memory_to_use*(total physical memory) as the memory chunk size.

Values accepted
---------------
Uint64. The default value is 500 with unit MB.

Example
-------
FLAGS_initial_cpu_memory_in_mb=100, if FLAGS_fraction_of_cpu_memory_to_use*(total physical memory) > 100MB, then allocator will pre-allocate 100MB when first allocation request raises, and re-allocate 100MB again when the pre-allocated memory is exhaustive.


FLAGS_initial_gpu_memory_in_mb
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


FLAGS_limit_of_tmp_allocation
*******************************************
(since 1.3)

The FLAGS_limit_of_tmp_allocation indicates the up limit of temporary_allocation size, the unit is byte. If the FLAGS_limit_of_tmp_allocation is -1, the size of temporary_allocation will not be limited.

Values accepted
---------------
Int64. The default value is -1.

Example
-------
FLAGS_limit_of_tmp_allocation=1024 will set the up limit of temporary_allocation size to 1024 bytes.


FLAGS_memory_fraction_of_eager_deletion
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


FLAGS_reallocate_gpu_memory_in_mb
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


FLAGS_times_excess_than_required_tmp_allocation
*******************************************
(since 1.3)

The FLAGS_times_excess_than_required_tmp_allocation indicates the max size the TemporaryAllocator can return. For Example
, if the required memory size is N, and FLAGS_times_excess_than_required_tmp_allocation is 2.0, the TemporaryAllocator will return the available allocation that the range of size is N ~ 2*N.

Values accepted
---------------
Int64. The default value is 2.

Example
-------
FLAGS_times_excess_than_required_tmp_allocation=1024 will set the max size of the TemporaryAllocator can return to 1024*N.


FLAGS_use_pinned_memory
*******************************************
(since 0.12.0)

Whether to use cpu pinned memory. If set, CPU allocator calls mlock to lock pages.

Values accepted
---------------
Bool. The default value is True.

Example
-------
FLAGS_use_pinned_memory=True would make the pages of allocated cpu memory lock.