.. _api_guide_memory_optimize_en:

###########
Memory Allocation and Optimization
###########

1. Memory Allocation Strategy
===========================

1.1. AutoGrowth Strategy
--------------------------

Since version 1.6+, PaddlePaddle supports the AutoGrowth strategy, which allocates memory on demand.
AutoGrowth strategy has been enabled by default in version 1.7+, making it convenient for users to
run multiple tasks on the same GPU card at the same time.

Because the native CUDA system calls :code:`cudaMalloc` and :code:`cudaFree` are synchronous operations,
which are very time-consuming, the AutoGrowth strategy will cache the allocated memory for subsequent allocation.
The specific methods are as follows:

- In the first few memory allocations, PaddlePaddle framework will call :code:`cudaMalloc` and allocate memory on demand. When releasing the allocated memory, it will not call :code:`cudaFree` to return the memory to GPU, but cache the memory inside the framework.

- In the subsequent allocations, PaddlePaddle framework will first check if there is a fit block (block size larger than the required memory size) in the cached memory. If there is, it will split the required memory from the fit block and return. Otherwise, it will call :code:`cudaMalloc` to allocate memory from GPU. The allocated memory are also cached when being released for subsequent allocation.

Therefore, the AutoGrowth strategy may slow the speed in the first few batches of model training,
but will not affect the speed in the subsequent training process.

1.2. Pre-Allocation Strategy
----------------

In addition to the AutoGrowth strategy, paddlepaddle also provides a Pre-Allocation strategy,
which is the default memory allocation strategy before paddlepaddle 1.7.

The Pre-Allocation strategy allocates a large size chunk at the first allocation, and the subsequent memory allocation is mostly obtained from the pre allocated memory chunk.
Among them, the chunk size is determined by the environment variable :code:`FLAGS_fraction_of_gpu_memory_to_use`, and the calculation formula of chunk size is:

.. code-block:: python

  chunk_size = FLAGS_fraction_of_gpu_memory_to_use * number of current available memory of a single GPU card

The default value of :code:`FLAGS_fraction_of_gpu_memory_to_use` is 0.92, that is, the framework will pre allocates
92% of the currently available memory of the GPU card.

The specific way of Pre-Allocation strategy to allocate GPU memory is:

- When allocating memory of requested_size,
    - If requested_size <= chunk_size, the framework will first allocate a memory chunk of chunk_size, then split a block of requested_size and return the block. Every subsequent memory allocation will be performed on the chunk.
    - If requested_size > chunk_size, the framework will call :code:`cudaMalloc` to allocate memory block of requested_size and return.

- When freeing memory of requested_size,
    - If free_size <= chunk_size, the framework will put the memory block back into the pre-allocated chunk, instead of returning back to GPU.
    - If free_size > chunk_size, the framework will call :code:`cudaFree` and return the memory back to GPU.

If there are other tasks on your GPU card that occupy the memory, you can appropriately decrease :code:`FLAGS_fraction_of_gpu_memory_to_use`
to ensure that the framework can pre-allocate the memory block of appropriate size, for example

.. code-block:: shell

  export FLAGS_fraction_of_gpu_memory_to_use=0.4 # Pre-allocate 40% memory of a single GPU card

If :code:`FLAGS_fraction_of_gpu_memory_to_use` is set to 0, the framework will call :code:`cudaMalloc` and :code:`cudaFree` every time the memory is allocated and released, which will seriously affect the performance and is not recommended. Only when you want to measure the actual memory usage of the network, you could set :code:`FLAGS_fraction_of_gpu_memory_to_use` to 0, and observe the memory usage of command nvidia-smi display.

1.3. Configuration of memory allocation strategy
-----------------------
Since version 1.6+, PaddlePaddle supports both the AutoGrowth strategy and the Pre-Allocation Strategy, and control the strategy used in framework by
the environment variable :code:`FLAGS_allocator_strategy`.

Use AutoGrowth strategy:

.. code-block:: shell

  export FLAGS_allocator_strategy=auto_growth # Use AutoGrowth strategy

Use Pre-Allocation strategy:

.. code-block:: shell

  export FLAGS_allocator_strategy=naive_best_fit # Use Pre-Allocation strategy

Plus, since version 1.7.2+, PaddlePaddle provides an environment variable :code:`FLAGS_gpu_memory_limit_mb`, which controls the maximum gpu memory limit that the process can allocate.
If it is equal to 0, there would be no limit and all gpu memory would be available to the process. If it is larger than 0, the process would raise out of memory error if the allocated
memory exceeds the limit even though there is available memory on the gpu card. The unit is MB and default value is 0.

2. Memory Optimization Strategy
===========================

Paddlepaddle provides several general memory optimization methods to optimize the memory usage of your network (including general memory and GPU memory).

2.1. GC Strategy: memory garbage eager collection
-------------------------

The principle of GC（Garbage Collection）is to release the memory space of useless variables eagerly during network running,
in order to save memory space. GC is suitable for training and inference using Executor or ParallelExecutor, but it is not suitable for C++ inference library.

**Since version 1.6+, GC Strategy is enabled by default.**

GC Strategy is controlled by 3 environment variable:


- :code:`FLAGS_eager_delete_tensor_gb`

Variable to enable GC, its data type is double. The default value is -1 in PaddlePaddle with version < 1.6,
and is 0 in PaddlePaddle with version >= 1.6. GC Strategy will cache a certain amount of memory garbage and release it uniformly.
:code:`FLAGS_eager_delete_tensor_gb` means the threshold of cached memory garbage, the unit of which is GB. **It is recommended to set** :code:`FLAGS_eager_delete_tensor_gb=0`.

If :code:`FLAGS_eager_delete_tensor_gb=0`, once there is memory garbage, it will be collected immediately to save memory.

If :code:`FLAGS_eager_delete_tensor_gb=1`, the memory garbage is collected when the cached amount of garbage reaches 1GB.

If :code:`FLAGS_eager_delete_tensor_gb<0`, GC Strategy is disabled.


- :code:`FLAGS_memory_fraction_of_eager_deletion`

Variable to control GC Strategy, its data type is double. The default value is 1, range [0,1]. It is only suitable for ParallelExecutor.
GC will sort the variables in descending order according to the memory space occupied by the variables,
and only collect the memory space of top :code:`FLAGS_memory_fraction_of_eager_deletion` variables.
**It is recommended to remain default value**, that is  :code:`FLAGS_memory_fraction_of_eager_deletion=1`.

If :code:`FLAGS_memory_fraction_of_eager_deletion=0.6`, top 60% variables will be collected.

If :code:`FLAGS_memory_fraction_of_eager_deletion=0`, no variable will be collected, GC Strategy is disabled.

If :code:`FLAGS_memory_fraction_of_eager_deletion=1`, all variables will be collected.


- :code:`FLAGS_fast_eager_deletion_mode`

Variable to enable fast GC Strategy, its type is bool. The default value is True, which means use fast GC Strategy.
Fast GC Strategy will collect the memory garbage immediately instead of waiting for CUDA Kernel finish. **It is recommended to remain default value**, that is  :code:`FLAGS_fast_eager_deletion_mode=True`.


2.2. Inplace Strategy: output reuses input inside operator
----------------------------------

The principle of Inplace strategy is that the output of some operators can reuses the memory space of input.
For example, the output and input of operator :code:`reshape` can reuse the same memory space.

Inplace Strategy is suitable for ParallelExecutor, which can be set through :code:`BuildStrategy`.
The Strategy is not suitable for Executor+Program or C++ inference library.

**Since version 1.6+, Inplace Strategy is enabled by default.**

The specific way of Inplace strategy is:

.. code-block:: python

    build_strategy = fluid.BuildStrategy()
    build_strategy.enable_inplace = True # Enable Inplace Strategy

    compiled_program = fluid.CompiledProgram(train_program, build_strategy=build_strategy)


In PaddlePaddle with version < 1.6, due to of some design problems, when the Inplace Strategy is enabled,
the variable in fetch_list in the subsequent :code:`exe.run` must be persistent.
That is, if you the variables you want to fetch are loss and acc, you must set:

.. code-block:: python

    loss.persistable = True
    acc.persistable = True


**Since version 1.6+, setting variables in fetch_list to persistable is not needed.**


3. Memory Optimization Best Practice
=======================

We recommend the best memory optimization strategy as:

- Enable GC strategy:set :code:`FLAGS_eager_delete_tensor_gb=0`.

- Enable Inplace strategy:set :code:`build_strategy.enable_inplace = True`, and set variables in fetch_list to persistable using :code:`var.persistable = True` when the version of PaddlePaddle < 1.6.

**Since version 1.6+, the above optimal strategy have been enabled by default and setting variables in fetch_list to persistable is not needed.**
