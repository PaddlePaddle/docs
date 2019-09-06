
存储管理
==================


FLAGS_allocator_strategy
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


FLAGS_eager_delete_scope
*******************************************
(始于0.12.0)

同步局域删除。设置后，它将降低GPU内存使用量，但同时也会减慢销毁变量的速度（性能损害约1％）。

取值范围
---------------
Bool型，缺省值为True。

示例
-------
FLAGS_eager_delete_scope=True - 同步局域删除。


FLAGS_eager_delete_tensor_gb
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


FLAGS_enable_inplace_whitelist
*******************************************
(始于1.4)

该flag用于调试，在某些ops中禁止内存原位复用。设置后，一些ops不会执行原位复用优化以节省内存。这些Ops包括：sigmoid, exp, relu, tanh, sqrt, ceil, floor, reciprocal, relu6, soft_relu, hard_sigmoid, batch_norm, batch_norm_grad, sum, sum_grad, scale, reshape, elementwise_add, and elementwise_add_grad。

取值范围
---------------
Bool型，缺省值为False。

示例
-------
FLAGS_enable_inplace_whitelist=True - 在特定op上禁止内存原位复用优化。


FLAGS_fast_eager_deletion_mode
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


FLAGS_fraction_of_gpu_memory_to_use
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


FLAGS_free_idle_memory
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


FLAGS_fuse_parameter_groups_size
*******************************************
(始于1.4.0)

FLAGS_fuse_parameter_groups_size表示每一组中参数的个数。缺省值是一个经验性的结果。如果fuse_parameter_groups_size为1，则表示组的大小和参数梯度的数目一致。 如果fuse_parameter_groups_size为-1，则表示只有一个组。缺省值为3，这只是一个经验值。

取值范围
---------------
Int32型，缺省值为3。

示例
-------
FLAGS_fuse_parameter_groups_size=3 - 将单组参数的梯度大小设为3。


FLAGS_fuse_parameter_memory_size
*******************************************
(始于1.5.0)

FLAGS_fuse_parameter_memory_size表示作为通信调用输入（例如NCCLAllReduce）的单组参数梯度的上限内存大小。默认值为-1.0，表示不根据memory_size设置组。单位是MB。

取值范围
---------------
Double型，缺省值为-1.0。

示例
-------
FLAGS_fuse_parameter_memory_size=16 - 将单组参数梯度的上限大小设为16MB。


FLAGS_init_allocated_mem
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


FLAGS_initial_cpu_memory_in_mb
*******************************************
(始于0.14.0)

初始PaddlePaddle分配器的CPU内存块大小，单位为MB。分配器将FLAGS_initial_cpu_memory_in_mb和FLAGS_fraction_of_cpu_memory_to_use*（总物理内存）的最小值作为内存块大小。

取值范围
---------------
Uint64型，缺省值为500，单位为MB。

示例
-------
FLAGS_initial_cpu_memory_in_mb=100 - 在FLAGS_fraction_of_cpu_memory_to_use*（总物理内存）大于100MB的情况下，首次提出分配请求时，分配器预先分配100MB内存，并在预分配的内存耗尽时再次分配100MB。


FLAGS_initial_gpu_memory_in_mb
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


FLAGS_limit_of_tmp_allocation
*******************************************
(始于1.3)

FLAGS_limit_of_tmp_allocation表示temporary_allocation大小的上限，单位为字节。如果FLAGS_limit_of_tmp_allocation为-1，temporary_allocation的大小将没有限制。

取值范围
---------------
Int64型，缺省值为-1。

示例
-------
FLAGS_limit_of_tmp_allocation=1024 - 将temporary_allocation大小的上限设为1024字节。


FLAGS_memory_fraction_of_eager_deletion
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


FLAGS_reallocate_gpu_memory_in_mb
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


FLAGS_times_excess_than_required_tmp_allocation
*******************************************
(始于1.3)

FLAGS_times_excess_than_required_tmp_allocation表示TemporaryAllocator可以返回的最大大小。例如，如果所需的内存大小为N，且times_excess_than_required_tmp_allocation为2.0，则TemporaryAllocator将返回大小范围为N~2*N的可用分配。

取值范围
---------------
Int64型，缺省值为2。

示例
-------
FLAGS_times_excess_than_required_tmp_allocation=1024 - 设置TemporaryAllocator可以返回的最大大小为1024*N。


FLAGS_use_pinned_memory
*******************************************
(始于0.12.0)

是否使用pinned memory。设为True后，CPU分配器将调用mlock来锁定内存页。

取值范围
---------------
Bool型，缺省值为True。

示例
-------
FLAGS_use_pinned_memory=True - 锁定分配的CPU内存页面。
