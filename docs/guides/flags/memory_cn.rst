
存储管理
==================


FLAGS_allocator_strategy
********************
(始于 1.2)

用于选择 PaddlePaddle 的分配器策略。

取值范围
---------------
String 型，['naive_best_fit', 'auto_growth']中的一个。缺省值如果编译 Paddle CMake 时使用-DON_INFER=ON 为'naive_best_fit'。
其他默认情况为'auto_growth'。PaddlePaddle pip 安装包的默认策略也是'auto_growth'

示例
--------
FLAGS_allocator_strategy=naive_best_fit - 使用预分配 best fit 分配器，PaddlePaddle 会先占用大多比例的可用内存/显存，在 Paddle 具体数据使用时分配，这种方式预占空间较大，但内存/显存碎片较少(比如能够支持模型的最大 batch size 会变大)。

FLAGS_allocator_strategy=auto_growth - 使用 auto growth 分配器。PaddlePaddle 会随着真实数据需要再占用内存/显存，但内存/显存可能会产生碎片（比如能够支持模型的最大 batch size 会变小）。


FLAGS_eager_delete_scope
*******************************************
(始于 0.12.0)

同步局域删除。设置后，它将降低 GPU 内存使用量，但同时也会减慢销毁变量的速度（性能损害约 1％）。

取值范围
---------------
Bool 型，缺省值为 True。

示例
-------
FLAGS_eager_delete_scope=True - 同步局域删除。


FLAGS_eager_delete_tensor_gb
*******************************************
(始于 1.0.0)

表示是否使用垃圾回收策略来优化网络的内存使用。如果 FLAGS_eager_delete_tensor_gb < 0，则禁用垃圾回收策略。如果 FLAGS_eager_delete_tensor_gb >= 0，则启用垃圾回收策略，并在运行网络时回收内存垃圾，这有利于节省内存使用量。它仅在您使用 Executor 运行程序、编译程序或使用并行数据编译程序时才有用。垃圾回收器直到垃圾的内存大小达到 FLAGS_eager_delete_tensor_gb GB 时才会释放内存垃圾。

取值范围
---------------
Double 型，单位为 GB，缺省值为 0.0。

示例
-------
FLAGS_eager_delete_tensor_gb=0.0 - 垃圾占用大小达到 0.0GB 时释放内存垃圾，即一旦出现垃圾则马上释放。

FLAGS_eager_delete_tensor_gb=1.0 - 垃圾占用内存大小达到 1.0GB 时释放内存垃圾。

FLAGS_eager_delete_tensor_gb=-1.0 - 禁用垃圾回收策略。

注意
-------
建议用户在训练大型网络时设置 FLAGS_eager_delete_tensor_gb=0.0 以启用垃圾回收策略。


FLAGS_fast_eager_deletion_mode
*******************************************
(始于 1.3)

是否使用快速垃圾回收策略。如果未设置，则在 CUDA 内核结束时释放 gpu 内存。否则 gpu 内存将在 CUDA 内核尚未结束的情况下被释放，从而使垃圾回收策略更快。仅在启用垃圾回收策略时有效。

取值范围
---------------
Bool 型，缺省值为 True。

示例
-------
FLAGS_fast_eager_deletion_mode=True - 启用快速垃圾回收策略。

FLAGS_fast_eager_deletion_mode=False - 禁用快速垃圾回收策略。


FLAGS_fraction_of_cpu_memory_to_use
*******************************************
(始于 1.2.0)

表示分配的内存块占 CPU 总内存大小的比例。将来的内存使用将从该内存块分配。 如果内存块没有足够的 cpu 内存，将从 cpu 请求分配与内存块相同大小的新的内存块，直到 cpu 没有足够的内存为止。

取值范围
---------------
Double 型，范围[0, 1]，表示初始分配的内存块占 CPU 内存的比例。缺省值为 1.0。

示例
-------
FLAGS_fraction_of_cpu_memory_to_use=0.1 - 分配总 CPU 内存大小的 10%作为初始 CPU 内存块。


FLAGS_fraction_of_cuda_pinned_memory_to_use
*******************************************
(始于 1.2.0)

表示分配的 CUDA Pinned 内存块占 CPU 总内存大小的比例。将来的 CUDA Pinned 内存使用将从该内存块分配。 如果内存块没有足够的 cpu 内存，将从 cpu 请求分配与内存块相同大小的新的内存块，直到 cpu 没有足够的内存为止。

取值范围
---------------
Double 型，范围[0, 1]，表示初始分配的内存块占 CPU 内存的比例。缺省值为 0.5。

示例
-------
FLAGS_fraction_of_cuda_pinned_memory_to_use=0.1 - 分配总 CPU 内存大小的 10%作为初始 CUDA Pinned 内存块。


FLAGS_fraction_of_gpu_memory_to_use
*******************************************
(始于 1.2.0)

表示分配的显存块占 GPU 总可用显存大小的比例。将来的显存使用将从该显存块分配。 如果显存块没有足够的 gpu 显存，将从 gpu 请求分配与显存块同样大小的新的显存块，直到 gpu 没有足够的显存为止。

取值范围
---------------
Double 型，范围[0, 1]，表示初始分配的显存块占 GPU 可用显存的比例。

示例
-------
FLAGS_fraction_of_gpu_memory_to_use=0.1 - 分配 GPU 总可用显存大小的 10%作为初始 GPU 显存块。

注意
-------
Windows 系列平台会将 FLAGS_fraction_of_gpu_memory_to_use 默认设为 0.5，Linux 则会默认设为 0.92。


FLAGS_fuse_parameter_groups_size
*******************************************
(始于 1.4.0)

FLAGS_fuse_parameter_groups_size 表示每一组中参数的个数。缺省值是一个经验性的结果。如果 fuse_parameter_groups_size 为 1，则表示组的大小和参数梯度的数目一致。 如果 fuse_parameter_groups_size 为-1，则表示只有一个组。缺省值为 3，这只是一个经验值。

取值范围
---------------
Int32 型，缺省值为 3。

示例
-------
FLAGS_fuse_parameter_groups_size=3 - 将单组参数的梯度大小设为 3。


FLAGS_fuse_parameter_memory_size
*******************************************
(始于 1.5.0)

FLAGS_fuse_parameter_memory_size 表示作为通信调用输入（例如 NCCLAllReduce）的单组参数梯度的上限内存大小。默认值为-1.0，表示不根据 memory_size 设置组。单位是 MB。

取值范围
---------------
Double 型，缺省值为-1.0。

示例
-------
FLAGS_fuse_parameter_memory_size=16 - 将单组参数梯度的上限大小设为 16MB。


FLAGS_init_allocated_mem
*******************************************
(始于 0.15.0)

是否对分配的内存进行非零值初始化。该 flag 用于调试，以防止某些 Ops 假定已分配的内存都是初始化为零的。

取值范围
---------------
Bool 型，缺省值为 False。

示例
-------
FLAGS_init_allocated_mem=True - 对分配的内存进行非零初始化。

FLAGS_init_allocated_mem=False - 不会对分配的内存进行非零初始化。


FLAGS_initial_cpu_memory_in_mb
*******************************************
(始于 0.14.0)

初始 PaddlePaddle 分配器的 CPU 内存块大小，单位为 MB。分配器将 FLAGS_initial_cpu_memory_in_mb 和 FLAGS_fraction_of_cpu_memory_to_use*（总物理内存）的最小值作为内存块大小。

取值范围
---------------
Uint64 型，缺省值为 500，单位为 MB。

示例
-------
FLAGS_initial_cpu_memory_in_mb=100 - 在 FLAGS_fraction_of_cpu_memory_to_use*（总物理内存）大于 100MB 的情况下，首次提出分配请求时，分配器预先分配 100MB 内存，并在预分配的内存耗尽时再次分配 100MB。


FLAGS_initial_gpu_memory_in_mb
*******************************************
(始于 1.4.0)

预分配一块指定大小的 GPU 显存块。之后的显存使用将从该显存块分配。如果显存块没有足够的显存，将从 GPU 请求大小为 FLAGS_reallocate_gpu_memory_in_mb 的显存块，直到 GPU 没有剩余显存为止。

取值范围
---------------
Uint64 型，大于 0，为初始 GPU 显存大小，单位为 MB。

示例
-------
FLAGS_initial_gpu_memory_in_mb=4096 - 分配 4GB 作为初始 GPU 显存块大小。

注意
-------
如果设置该 flag，则 FLAGS_fraction_of_gpu_memory_to_use 设置的显存大小将被该 flag 覆盖。PaddlePaddle 将用该 flag 指定的值分配初始 GPU 显存。
如果未设置该 flag，即 flag 默认值为 0 时，会关闭此显存策略。PaddlePaddle 会使用 FLAGS_fraction_of_gpu_memory_to_use 的策略来分配初始显存块。


FLAGS_memory_fraction_of_eager_deletion
*******************************************
(始于 1.4)

垃圾回收策略释放变量的内存大小百分比。如果 FLAGS_memory_fraction_of_eager_deletion = 1.0，则将释放网络中的所有临时变量。如果 FLAGS_memory_fraction_of_eager_deletion = 0.0，则不会释放网络中的任何临时变量。如果 0.0<FLAGS_memory_fraction_of_eager_deletion<1.0，则所有临时变量将根据其内存大小降序排序，并且仅
释放具有最大内存大小的 FLAGS_memory_fraction_of_eager_deletion 比例的变量。该 flag 仅在运行并行数据编译程序时有效。

取值范围
---------------
Double 型，范围为[0.0, 1.0]，缺省值为 1.0。

示例
-------
FLAGS_memory_fraction_of_eager_deletion=0 - 保留所有临时变量，也就是禁用垃圾回收策略。

FLAGS_memory_fraction_of_eager_deletion=1 - 释放所有临时变量。

FLAGS_memory_fraction_of_eager_deletion=0.5 - 仅释放 50%比例的占用内存最多的变量。


FLAGS_reallocate_gpu_memory_in_mb
*******************************************
(始于 1.4.0)

如果耗尽了分配的 GPU 显存块，则重新分配额外的 GPU 显存块。

取值范围
---------------
Int64 型，大于 0，为重新分配的显存大小，单位为 MB。

示例
-------
FLAGS_reallocate_gpu_memory_in_mb=1024 - 如果耗尽了分配的 GPU 显存块，重新分配 1GB。

注意
-------
如果设置了该 flag，则 FLAGS_fraction_of_gpu_memory_to_use 设置的显存大小将被该 flag 覆盖，PaddlePaddle 将用该 flag 指定的值重分配额外 GPU 显存。
如果未设置该 flag，即 flag 默认值为 0 时，会关闭此显存策略。PaddlePaddle 会使用 FLAGS_fraction_of_gpu_memory_to_use 的策略来重新分配额外显存。

FLAGS_use_pinned_memory
*******************************************
(始于 0.12.0)

是否使用 pinned memory。设为 True 后，CPU 分配器将调用 mlock 来锁定内存页。

取值范围
---------------
Bool 型，缺省值为 True。

示例
-------
FLAGS_use_pinned_memory=True - 锁定分配的 CPU 内存页面。
