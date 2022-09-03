.. _api_guide_memory_optimize:

###########
存储分配与优化
###########

1. PaddlePaddle 的显存分配策略
===========================

1.1. 显存自增长 AutoGrowth 策略
--------------------------
自 1.6+的版本起，PaddlePaddle 支持显存自增长 AutoGrowth 策略，按需分配显存，且已于 1.7+版本中默认开启，方便用户在同一张 GPU 卡上同时运行多个任务。

由于原生的 CUDA 系统调用 :code:`cudaMalloc` 和 :code:`cudaFree` 均是同步操作，非常耗时。
因此显存自增长 AutoGrowth 策略会缓存已分配到的显存，供后续分配使用，具体方式为：

- 在前几次显存分配时，框架会调用 :code:`cudaMalloc` 按需分配，但释放时不会调用 :code:`cudaFree` 返回给 GPU，而是在框架内部缓存起来。

- 在随后的显存分配时，框架会首先检查缓存的显存中是否有合适的块，若有则从中分割出所需的显存空间返回，否则才调用 :code:`cudaMalloc` 直接从 GPU 中分配。随后的显存释放亦会缓存起来供后续分配使用。

因此，显存自增长 AutoGrowth 策略会在前几个 batch 训练时分配较慢（因为频繁调用 :code:`cudaMalloc` ），在随后训练过程中基本不会影响模型训练速度。

1.2. 显存预分配策略
----------------

除了显存自增长 AutoGrowth 策略以外，PaddlePaddle 还提供了显存预分配策略。显存预分配策略是 PaddlePaddle 1.7 版本前的默认显存分配策略。

显存预分配策略会在第一次分配时分配很大 chunk_size 的显存块，随后的显存分配大多从预分配的显存块中切分获得。
其中，chunk_size 由环境变量 :code:`FLAGS_fraction_of_gpu_memory_to_use` 确定，chunk_size 的计算公式为：

.. code-block:: python

  chunk_size = FLAGS_fraction_of_gpu_memory_to_use * 单张 GPU 卡的当前可用显存值

:code:`FLAGS_fraction_of_gpu_memory_to_use` 的默认值为 0.92，即框架预先分配显卡 92%的当前可用显存值。

显存预分配策略分配显存的具体方式为：

- 在分配 requested_size 大小的显存时，
    - 若 requested_size <= chunk_size，则框架会预先分配 chunk_size 大小的显存池 chunk，并从 chunk 中分出 requested_size 大小的块返回。之后每次申请显存都会从 chunk 中分配。
    - 若 requested_size > chunk_size，则框架会直接调用 :code:`cudaMalloc` 分配 requested_size 大小的显存返回。

- 在释放 free_size 大小的显存时，
    - 若 free_size <= chunk_size，则框架会将该显存放回预分配的 chunk 中，而不是直接返回给 CUDA。
    - 若 free_size > chunk_size，则框架会直接调用 :code:`cudaFree` 将显存返回给 CUDA。

若你的 GPU 卡上有其他任务占用显存，你可以适当将 :code:`FLAGS_fraction_of_gpu_memory_to_use` 减少，保证框架能预分配到合适的显存块，例如：

.. code-block:: shell

  export FLAGS_fraction_of_gpu_memory_to_use=0.4 # 预先 40%的 GPU 显存

若 :code:`FLAGS_fraction_of_gpu_memory_to_use` 设为 0，则每次显存分配和释放均会调用 :code:`cudaMalloc` 和 :code:`cudaFree` ，会严重影响性能，不建议你使用。
只有当你想测量网络的实际显存占用量时，你可以设置 :code:`FLAGS_fraction_of_gpu_memory_to_use` 为 0，观察 nvidia-smi 显示的显存占用情况。

1.3. 显存分配策略的选择方式
-----------------------
自 1.6+版本起，PaddlePaddle 同时支持显存自增长 AutoGrowth 策略和显存预分配策略，并通过环境变量 :code:`FLAGS_allocator_strategy` 控制。

选择显存自增长 AutoGrowth 的方式为：

.. code-block:: shell

  export FLAGS_allocator_strategy=auto_growth # 选择显存自增长 AutoGrowth 策略

选择显存预分配策略的方式为：

.. code-block:: shell

  export FLAGS_allocator_strategy=naive_best_fit # 选择显存预分配策略

此外，自 1.7.2+版本起，PaddlePaddle 提供了环境变量 :code:`FLAGS_gpu_memory_limit_mb` ，用于控制单个任务进程可分配的最大显存，单位是 MB。默认值是 0，表示没有限制，可分配全部显存。如果设置为大于 0 的值，则会在分配的显存超过限制时报错，即使此时系统还存在空闲的显存空间。

2. PaddlePaddle 的存储优化策略
===========================

PaddlePaddle 提供了多种通用存储优化方法，优化你的网络的存储占用（包括显存和内存)。

2.1. GC 策略: 存储垃圾及时回收
-------------------------

GC（Garbage Collection）的原理是在网络运行阶段及时释放无用变量的存储空间，达到节省存储空间的目的。GC 适用于使用 Executor，ParallelExecutor 做模型训练/预测的场合，但不适用于 C++预测库接口。

**GC 策略已于 1.6+版本中默认开启。**

GC 策略由三个环境变量控制：


- :code:`FLAGS_eager_delete_tensor_gb`

GC 策略的使能开关，double 类型，在<1.6 的版本中默认值为-1，在 1.6+版本中默认值为 0。GC 策略会积攒一定大小的存储垃圾后再统一释放，:code:`FLAGS_eager_delete_tensor_gb` 控制的是存储垃圾的阈值，单位是 GB。**建议用户设置** :code:`FLAGS_eager_delete_tensor_gb=0` 。

若 :code:`FLAGS_eager_delete_tensor_gb=0` ，则一旦有存储垃圾则马上回收，最为节省存储空间。

若 :code:`FLAGS_eager_delete_tensor_gb=1` ，则存储垃圾积攒到 1G 后才触发回收。

若 :code:`FLAGS_eager_delete_tensor_gb<0` ，则 GC 策略关闭。


- :code:`FLAGS_memory_fraction_of_eager_deletion`

GC 策略的调节 flag，double 类型，默认值为 1，范围为[0,1]，仅适用于使用 ParallelExecutor 或 CompiledProgram+with_data_parallel 的场合。
GC 内部会根据变量占用的存储空间大小，对变量进行降序排列，且仅回收前 :code:`FLAGS_memory_fraction_of_eager_deletion` 大的变量的存储空间。**建议用户维持默认值**，即 :code:`FLAGS_memory_fraction_of_eager_deletion=1` 。

若 :code:`FLAGS_memory_fraction_of_eager_deletion=0.6` ，则表示仅回收存储占用 60%大的变量的存储空间。

若 :code:`FLAGS_memory_fraction_of_eager_deletion=0` ，则表示不回收任何变量的存储空间，GC 策略关闭。

若 :code:`FLAGS_memory_fraction_of_eager_deletion=1` ，则表示回收所有变量的存储空间。


- :code:`FLAGS_fast_eager_deletion_mode`

快速 GC 策略的开关，bool 类型，默认值为 True，表示使用快速 GC 策略。快速 GC 策略会不等待 CUDA Kernel 结束直接释放显存。**建议用户维持默认值**，即 :code:`FLAGS_fast_eager_deletion_mode=True` 。


2.2. Inplace 策略: Op 内部的输出复用输入
----------------------------------

Inplace 策略的原理是 Op 的输出复用 Op 输入的存储空间。例如，reshape 操作的输出和输入可复用同一片存储空间。

Inplace 策略适用于使用 ParallelExecutor 或 CompiledProgram+with_data_parallel 的场合，通过 :code:`BuildStrategy` 设置。此策略不支持使用 Executor+Program 做单卡训练、使用 C++预测库接口等场合。

**Inplace 策略已于 1.6+版本中默认开启。**

具体方式为:

.. code-block:: python

    build_strategy = fluid.BuildStrategy()
    build_strategy.enable_inplace = True # 开启 Inplace 策略

    compiled_program = fluid.CompiledProgram(train_program)
                              .with_data_parallel(loss_name=loss.name, build_strategy=build_strategy)


在<1.6 的版本中，由于设计上的一些问题，在开启 Inplace 策略后，必须保证后续 exe.run 中 fetch_list 的变量是 persistable 的，即假如你后续需要 fetch 的变量为 loss 和 acc，则必须设置：

.. code-block:: python

    loss.persistable = True
    acc.persistable = True


**在 1.6+的版本中，无需设置 fetch 变量为 persistable。**


3. 存储优化 Best Practice
=======================

我们推荐你的最佳存储优化策略为：

- 开启 GC 策略：设置 :code:`FLAGS_eager_delete_tensor_gb=0` 。

- 开启 Inplace 策略：设置 :code:`build_strategy.enable_inplace = True` ，并在<1.6 版本中设置 fetch_list 中的 :code:`var.persistable = True` 。

**在 1.6+的版本中，上述最佳策略均已默认打开，无需手动配置，亦无需设置 fetch_list 变量为 persistable。**
