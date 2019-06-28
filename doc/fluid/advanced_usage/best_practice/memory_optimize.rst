.. _api_guide_memory_optimize:

###########
显存分配与优化
###########

PaddlePaddle的显存分配策略
========================

由于原生的CUDA系统调用 :code:`cudaMalloc` 和 :code:`cudaFree` 均是同步操作，非常耗时。因此与许多框架类似，PaddlePaddle采用了显存预分配的策略加速显存分配。具体方式为：

- 在分配requested_size大小的显存时，
    - 若requested_size <= chunk_size，则框架会预先分配chunk_size大小的显存池chunk，并从chunk中分出requested_size大小的块返回。之后每次申请显存都会从chunk中分配。
    - 若requested_size > chunk_size，则框架会直接调用 :code:`cudaMalloc` 分配requested_size大小的显存返回。

- 在释放free_size大小的显存时，
    - 若free_size <= chunk_size，则框架会将该显存放回预分配的chunk中，而不是直接返回给CUDA。
    - 若free_size > chunk_size，则框架会直接调用 :code:`cudaFree` 将显存返回给CUDA。

上述的chunk_size由环境变量 :code:`FLAGS_fraction_of_gpu_memory_to_use` 确定，chunk_size的计算公式为：

.. code-block:: python

  chunk_size = FLAGS_fraction_of_gpu_memory_to_use * 单张GPU卡的总显存


:code:`FLAGS_fraction_of_gpu_memory_to_use` 的默认值为0.92，即框架预先分配显卡92%的显存。

若你的GPU卡上有其他任务占用显存，你可以适当将 :code:`FLAGS_fraction_of_gpu_memory_to_use` 减少，保证框架能预分配到合适的chunk，例如：

.. code-block:: shell

  export FLAGS_fraction_of_gpu_memory_to_use=0.4 # 预先40%的GPU显存

若 :code:`FLAGS_fraction_of_gpu_memory_to_use` 设为0，则每次显存分配和释放均会调用 :code:`cudaMalloc` 和 :code:`cudaFree` ，会严重影响性能，不建议你使用。
只有当你想测量网络的实际显存占用量时，你可以设置 :code:`FLAGS_fraction_of_gpu_memory_to_use` 为0，观察nvidia-smi显示的显存占用情况。

PaddlePaddle的显存优化策略
========================

PaddlePaddle提供了多种通用显存优化方法，优化你的网络的显存占用。

GC策略: 显存垃圾及时回收
====================

GC（Garbage Collection）的原理是在网络运行阶段及时释放无用变量的显存空间，达到节省显存的目的。GC适用于使用Executor，ParallelExecutor做模型训练/预测的场合。

GC策略由三个环境变量控制：

- :code:`FLAGS_eager_delete_tensor_gb`

GC策略的使能开关，double类型，默认值为-1。GC策略会积攒一定大小的显存垃圾后再统一释放，:code:`FLAGS_eager_delete_tensor_gb` 控制的是显存垃圾的阈值，单位是GB。**建议用户设置** :code:`FLAGS_eager_delete_tensor_gb=0` 。

若 :code:`FLAGS_eager_delete_tensor_gb=0` ，则一旦有显存垃圾则马上回收，最为节省显存。

若 :code:`FLAGS_eager_delete_tensor_gb=1` ，则显存垃圾积攒到1G后才触发回收。

若 :code:`FLAGS_eager_delete_tensor_gb<0` ，则GC策略关闭。

- :code:`FLAGS_memory_fraction_of_eager_deletion`

GC策略的调节flag，double类型，默认值为1，范围为[0,1]，仅适用于使用ParallelExecutor或CompiledProgram+with_data_parallel的场合。
GC内部会根据变量占用的显存大小，对变量进行降序排列，且仅回收前 :code:`FLAGS_memory_fraction_of_eager_deletion` 大的变量显存。**建议用户维持默认值**，即 :code:`FLAGS_memory_fraction_of_eager_deletion=1` 。

若 :code:`FLAGS_memory_fraction_of_eager_deletion=0.6` ，则表示仅回收显存占用60%大的变量显存。

若 :code:`FLAGS_memory_fraction_of_eager_deletion=0` ，则表示不回收任何变量的显存，GC策略关闭。

若 :code:`FLAGS_memory_fraction_of_eager_deletion=1` ，则表示回收所有变量的显存。

- :code:`FLAGS_fast_eager_deletion_mode`

快速GC策略的开关，bool类型，默认值为True，表示使用快速GC策略。快速GC策略会不等待CUDA Kernel结束直接释放显存。**建议用户维持默认值**，即 :code:`FLAGS_fast_eager_deletion_mode=True` 。

Inplace策略: Op内部的输出复用输入
=============================

Inplace策略的原理是Op的输出复用Op输入的显存空间。例如，reshape操作的输出和输入可复用同一片显存空间。

Inplace策略适用于使用ParallelExecutor或CompiledProgram+with_data_parallel的场合，通过 :code:`BuildStrategy` 设置。

具体方式为:

.. code-block:: python

    build_strategy = fluid.BuildStrategy()
    build_strategy.enable_inplace = True # 开启Inplace策略

    compiled_program = fluid.CompiledProgram(train_program)
                              .with_data_parallel(loss_name=loss.name, build_strategy=build_strategy)

由于目前设计上的一些问题，在开启Inplace策略后，必须保证后续exe.run中fetch_list的变量是persistable的，即假如你后续需要fetch的变量为loss和acc，则必须设置：

.. code-block:: python

    loss.persistable = True
    acc.persistable = True

MemoryOptimize策略: 跨Op间的显存复用（不推荐）
========================================

MemoryOptimize策略的原理是当前Op的输出变量复用前继Op的无用变量空间。由于MemoryOptimize策略会延长显存空间的生命周期，这部分复用的显存可能无法及时释放，导致显存峰值升高，因此不建议用户使用该开关。

由于历史原因，PaddlePaddle提供了2个MemoryOptimize接口：

- :code:`BuildStrategy` 中的 :code:`memory_optimize` ：设置 :code:`build_strategy.memory_optimize=True` 开启MemoryOptimize策略。

- :code:`fluid.memory_optimize()` 接口：**该接口已废弃，不建议用户使用！**

与Inplace策略相同，开启MemoryOptimize策略时同样要保证后续exe.run中fetch_list的变量是persistable的。

显存优化Best Practice
====================

我们推荐你的最佳显存优化策略为：

- 开启GC策略：设置 :code:`FLAGS_eager_delete_tensor_gb=0` 。

- 开启Inplace策略：设置 :code:`build_strategy.enable_inplace = True` ，并设置fetch_list中的 :code:`var.persistable = True` 。

