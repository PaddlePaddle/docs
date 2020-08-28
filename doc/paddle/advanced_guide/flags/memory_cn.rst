
存储管理
==================


FLAGS_allocator_strategy
********************
(始于1.2)

用于选择PaddlePaddle的分配器策略。

取值范围
---------------
String型，['naive_best_fit', 'auto_growth']中的一个。缺省值如果编译Paddle CMake时使用-DON_INFER=ON为'naive_best_fit'。
其他默认情况为'auto_growth'。PaddlePaddle pip安装包的默认策略也是'auto_growth'

示例
--------
FLAGS_allocator_strategy=naive_best_fit - 使用预分配best fit分配器，PaddlePaddle会先占用大多比例的可用内存/显存，在Paddle具体数据使用时分配，这种方式预占空间较大，但内存/显存碎片较少(比如能够支持模型的最大batch size会变大)。

FLAGS_allocator_strategy=auto_growth - 使用auto growth分配器。PaddlePaddle会随着真实数据需要再占用内存/显存，但内存/显存可能会产生碎片（比如能够支持模型的最大batch size会变小）。


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

表示是否使用垃圾回收策略来优化网络的内存使用。如果FLAGS_eager_delete_tensor_gb < 0，则禁用垃圾回收策略。如果FLAGS_eager_delete_tensor_gb >= 0，则启用垃圾回收策略，并在运行网络时回收内存垃圾，这有利于节省内存使用量。它仅在您使用Executor运行程序、编译程序或使用并行数据编译程序时才有用。垃圾回收器直到垃圾的内存大小达到FLAGS_eager_delete_tensor_gb GB时才会释放内存垃圾。

取值范围
---------------
Double型，单位为GB，缺省值为0.0。

示例
-------
FLAGS_eager_delete_tensor_gb=0.0 - 垃圾占用大小达到0.0GB时释放内存垃圾，即一旦出现垃圾则马上释放。

FLAGS_eager_delete_tensor_gb=1.0 - 垃圾占用内存大小达到1.0GB时释放内存垃圾。

FLAGS_eager_delete_tensor_gb=-1.0 - 禁用垃圾回收策略。    

注意
-------
建议用户在训练大型网络时设置FLAGS_eager_delete_tensor_gb=0.0以启用垃圾回收策略。


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


FLAGS_fraction_of_cpu_memory_to_use
*******************************************
(始于1.2.0)

表示分配的内存块占CPU总内存大小的比例。将来的内存使用将从该内存块分配。 如果内存块没有足够的cpu内存，将从cpu请求分配与内存块相同大小的新的内存块，直到cpu没有足够的内存为止。

取值范围
---------------
Double型，范围[0, 1]，表示初始分配的内存块占CPU内存的比例。缺省值为1.0。

示例
-------
FLAGS_fraction_of_cpu_memory_to_use=0.1 - 分配总CPU内存大小的10%作为初始CPU内存块。


FLAGS_fraction_of_cuda_pinned_memory_to_use
*******************************************
(始于1.2.0)

表示分配的CUDA Pinned内存块占CPU总内存大小的比例。将来的CUDA Pinned内存使用将从该内存块分配。 如果内存块没有足够的cpu内存，将从cpu请求分配与内存块相同大小的新的内存块，直到cpu没有足够的内存为止。

取值范围
---------------
Double型，范围[0, 1]，表示初始分配的内存块占CPU内存的比例。缺省值为0.5。

示例
-------
FLAGS_fraction_of_cuda_pinned_memory_to_use=0.1 - 分配总CPU内存大小的10%作为初始CUDA Pinned内存块。


FLAGS_fraction_of_gpu_memory_to_use
*******************************************
(始于1.2.0)

表示分配的显存块占GPU总可用显存大小的比例。将来的显存使用将从该显存块分配。 如果显存块没有足够的gpu显存，将从gpu请求分配与显存块同样大小的新的显存块，直到gpu没有足够的显存为止。

取值范围
---------------
Double型，范围[0, 1]，表示初始分配的显存块占GPU可用显存的比例。

示例
-------
FLAGS_fraction_of_gpu_memory_to_use=0.1 - 分配GPU总可用显存大小的10%作为初始GPU显存块。

注意
-------
Windows系列平台会将FLAGS_fraction_of_gpu_memory_to_use默认设为0.5，Linux则会默认设为0.92。


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

预分配一块指定大小的GPU显存块。之后的显存使用将从该显存块分配。如果显存块没有足够的显存，将从GPU请求大小为FLAGS_reallocate_gpu_memory_in_mb的显存块，直到GPU没有剩余显存为止。

取值范围
---------------
Uint64型，大于0，为初始GPU显存大小，单位为MB。

示例
-------
FLAGS_initial_gpu_memory_in_mb=4096 - 分配4GB作为初始GPU显存块大小。

注意
-------
如果设置该flag，则FLAGS_fraction_of_gpu_memory_to_use设置的显存大小将被该flag覆盖。PaddlePaddle将用该flag指定的值分配初始GPU显存。
如果未设置该flag，即flag默认值为0时，会关闭此显存策略。PaddlePaddle会使用FLAGS_fraction_of_gpu_memory_to_use的策略来分配初始显存块。


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

如果耗尽了分配的GPU显存块，则重新分配额外的GPU显存块。

取值范围
---------------
Int64型，大于0，为重新分配的显存大小，单位为MB。

示例
-------
FLAGS_reallocate_gpu_memory_in_mb=1024 - 如果耗尽了分配的GPU显存块，重新分配1GB。

注意
-------
如果设置了该flag，则FLAGS_fraction_of_gpu_memory_to_use设置的显存大小将被该flag覆盖，PaddlePaddle将用该flag指定的值重分配额外GPU显存。
如果未设置该flag，即flag默认值为0时，会关闭此显存策略。PaddlePaddle会使用FLAGS_fraction_of_gpu_memory_to_use的策略来重新分配额外显存。

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
