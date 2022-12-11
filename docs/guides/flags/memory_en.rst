
memory management
==================


FLAGS_allocator_strategy
**************************************
(since 1.2)

Use to choose allocator strategy of PaddlePaddle.

Values accepted
---------------
String, enum in ['naive_best_fit', 'auto_growth']. The default value will be 'naive_best_fit' if users compile PaddlePaddle with -DON_INFER=ON CMake flag, otherwise is 'auto_growth'. The default PaddlePaddle pip package uses 'auto_growth'.

Example
--------
FLAGS_allocator_strategy=naive_best_fit would use the pre-allocated best fit allocator. 'naive_best_fit' strategy would occupy almost all GPU memory by default but leads to less memory fragmentation (i.e., maximum batch size of models may be larger).

FLAGS_allocator_strategy=auto_growth would use the auto growth allocator. 'auto_growth' strategy would allocate GPU memory on demand but may lead to more memory fragmentation (i.e., maximum batch size of models may be smaller).



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

Whether to use garbage collection strategy to optimize the memory usage of network. If FLAGS_eager_delete_tensor_gb < 0, garbage collection strategy is disabled. If FLAGS_eager_delete_tensor_gb >= 0, garbage collection strategy would be enabled, and collect memory garbages when running network, which is beneficial to saving memory usage. It is only useful when you use Executor to run program, or compile program, or compile program with data parallel. Garbage collector would not release memory garbages until the memory size of garbages reaches FLAGS_eager_delete_tensor_gb GB.

Values accepted
---------------
Double, in GB unit. The default value is 0.0.

Example
-------
FLAGS_eager_delete_tensor_gb=0.0 would make memory garbage release till the memory size of garbages reaches 0.0GB, i.e., release immediately once there is any garbage.

FLAGS_eager_delete_tensor_gb=1.0 would make memory garbage release till the memory size of garbages reaches 1.0GB.

FLAGS_eager_delete_tensor_gb=-1.0 would disable garbage collection strategy.

Note
-------
It is recommended that users enable garbage collection strategy by setting FLAGS_eager_delete_tensor_gb=0.0 when training large network.


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

FLAGS_fraction_of_cpu_memory_to_use
*******************************************
(since 1.2.0)

Allocate a chunk of cpu memory that is this fraction of the total cpu memory size. Future memory usage will be allocated from the chunk. If the chunk doesn't have enough cpu memory, additional chunks of the same size will be requested from cpu until the cpu has no memory left for another chunk.

Values accepted
---------------
Double value in range [0, 1] which is the initial CPU memory percentage. The default value is 1.0.

Example
-------
FLAGS_fraction_of_cpu_memory_to_use=0.1 will allocate 10% total cpu memory size as initial CPU chunk.


FLAGS_fraction_of_cuda_pinned_memory_to_use
*******************************************
(since 1.2.0)

Allocate a chunk of CUDA pinned memory that is this fraction of the total cpu memory size. Future memory usage will be allocated from the chunk. If the chunk doesn't have enough cpu memory, additional chunks of the same size will be requested from cpu until the cpu has no memory left for another chunk.

Values accepted
---------------
Double value in range [0, 1] which is the initial CUDA pinned memory percentage. The default value is 0.5.

Example
-------
FLAGS_fraction_of_cuda_pinned_memory_to_use=0.1 will allocate 10% total cpu memory size as initial CUDA Pinned chunk.


FLAGS_fraction_of_gpu_memory_to_use
*******************************************
(since 1.2.0)

Allocate a chunk of gpu memory that is this fraction of the available gpu memory size. Future memory usage will be allocated from the chunk. If the chunk doesn't have enough gpu memory, additional chunks of the same size will be requested from gpu until the gpu has no memory left for another chunk.

Values accepted
---------------
Double value in range [0, 1] which is the initial GPU memory percentage.

Example
-------
FLAGS_fraction_of_gpu_memory_to_use=0.1 will allocate 10% available gpu memory size as initial GPU chunk.

Note
-------
Windows series platform will set FLAGS_fraction_of_gpu_memory_to_use to 0.5 by default.
Linux will set FLAGS_fraction_of_gpu_memory_to_use to 0.92 by default.


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

Allocate a chunk of GPU memory whose byte size is specified by the flag. Future memory usage will be allocated from the chunk. If the chunk doesn't have enough GPU memory, additional chunks of the GPU memory will be requested from GPU with size specified by FLAGS_reallocate_gpu_memory_in_mb until the GPU has no memory left for the additional chunk.

Values accepted
---------------
Uint64 value greater than 0 which is the initial GPU memory size in MB.

Example
-------
FLAGS_initial_gpu_memory_in_mb=4096 will allocate 4 GB as initial GPU chunk.

Note
-------
If you set this flag, the memory size set by FLAGS_fraction_of_gpu_memory_to_use will be overrided by this flag, PaddlePaddle will allocate the initial gpu memory with size specified by this flag.
If you don't set this flag, the dafault value 0 will disable this GPU memory strategy. PaddlePaddle will use FLAGS_fraction_of_gpu_memory_to_use to allocate the initial GPU chunk.



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
Int64 value greater than 0 in MB which is the re-allocated GPU memory size in MB

Example
-------
FLAGS_reallocate_gpu_memory_in_mb=1024 will re-allocate 1 GB if run out of GPU memory chunk.

Note
-------
If this flag is set, the memory size set by FLAGS_fraction_of_gpu_memory_to_use will be overrided by this flag, PaddlePaddle will re-allocate the gpu memory with size specified by this flag.
If you don't set this flag, the dafault value 0 will disable this GPU memory strategy. PaddlePaddle will use FLAGS_fraction_of_gpu_memory_to_use to re-allocate GPU memory.


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
