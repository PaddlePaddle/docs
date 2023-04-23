.. _api_guide_parallel_executor:

#####
数据并行执行引擎
#####


:code:`ParallelExecutor` 是以数据并行的方式在多个节点上分别执行 :code:`Program` 的执行器。用户可以通过 Python 脚本驱动 :code:`ParallelExecutor` 执行， :code:`ParallelExecutor` 的执行过程：

- 首先根据 :code:`Program` 、 :code:`GPU` 卡的数目（或者 :code:`CPU` 的核数）以及 :ref:`cn_api_fluid_BuildStrategy` 构建 :code:`SSA Graph` 和一个线程池;
- 执行过程中，根据 Op 的输入是否 Ready 决定是否执行该 Op，这样可以使没有相互依赖的多个 Op 可在线程池中并行执行；

:code:`ParallelExecutor` 在构造时需要指定当前 :code:`Program` 的设备类型， :code:`GPU` 或者 :code:`CPU` ：

* 使用 :code:`GPU` 执行： :code:`ParallelExecutor` 会自动检测当前机器可以使用 :code:`GPU` 的个数，并在每个 :code:`GPU` 上分别执行 :code:`Program` ，用户也可以通过设置 :code:`CUDA_VISIBLE_DEVICES` 环境变量来指定执行器可使用的 :code:`GPU` ；
* 使用 :code:`CPU` 多线程执行：:code:`ParallelExecutor` 会自动检测当前机器可利用的 :code:`CPU` 核数，并将 :code:`CPU` 核数作为执行器中线程的个数，每个线程分别执行 :code:`Program` ，用户也可以通过设置 :code:`CPU_NUM` 环境变量来指定当前训练使用的线程个数。

:code:`ParallelExecutor` 支持模型训练和模型预测：

* 模型训练： :code:`ParallelExecutor` 在执行过程中对多个节点上的参数梯度进行聚合，然后进行参数的更新；
* 模型预测： :code:`ParallelExecutor` 在执行过程中各个节点独立运行当前的  :code:`Program` ；

:code:`ParallelExecutor` 在模型训练时支持两种模式的梯度聚合, :code:`AllReduce` 和 :code:`Reduce` ：

* :code:`AllReduce` 模式下， :code:`ParallelExecutor` 调用 AllReduce 操作使多个节点上参数梯度完全相等，然后各个节点独立进行参数的更新；
* :code:`Reduce` 模式下， :code:`ParallelExecutor` 会预先将所有参数的更新分派到不同的节点上，在执行过程中 :code:`ParallelExecutor` 调用 Reduce 操作将参数梯度在预先指定的节点上进行聚合，并进行参数更新，最后调用 Broadcast 操作将更新后的参数发送到其他节点。

这两种模式通过 :code:`build_strategy` 来指定，使用方法，请参考 :ref:`cn_api_fluid_BuildStrategy` 。

**注意** ：如果在 Reduce 模式下使用 :code:`CPU` 多线程执行 :code:`Program` ， :code:`Program` 的参数在多个线程间是共享的，在某些模型上，Reduce 模式可以大幅节省内存。

鉴于模型的执行速率和模型结构及执行器的执行策略有关，:code:`ParallelExecutor` 允许你修改执行器的相关参数，例如线程池的规模( :code:`num_threads` )、为清除临时变量 :code:`num_iteration_per_drop_scope` 需要进行的循环次数。更多信息请参照 :ref:`cn_api_fluid_ExecutionStrategy` 。


.. code-block:: python

    # 注释：
    #   - 如果你想在 ParallelExecutor 中指定用于运行的 GPU 卡，需要在环境中定义
    #     CUDA_VISIBLE_DEVICES
    #   - 如果你想在 ParallelExecutor 中使用多 CPU 来运行程序，需要在环境中定义
    #     CPU_NUM
    # 首先创建 Executor。
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    # 运行启动程序仅一次。
    exe.run(fluid.default_startup_program())
    # 定义 train_exe 和 test_exe
    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = dev_count * 4 # the size of thread pool.
    build_strategy = fluid.BuildStrategy()
    build_strategy.memory_optimize = True if memory_opt else False
    train_exe = fluid.ParallelExecutor(use_cuda=use_cuda,
                                       main_program=train_program,
                                       build_strategy=build_strategy,
                                       exec_strategy=exec_strategy,
                                       loss_name=loss.name)
    # 注释：对于 test_exe，loss_name 是不必要的。
    test_exe = fluid.ParallelExecutor(use_cuda=True,
                                      main_program=test_program,
                                      build_strategy=build_strategy,
                                      exec_strategy=exec_strategy,
                                      share_vars_from=train_exe)
    train_loss, = train_exe.run(fetch_list=[loss.name], feed=feed_dict)
    test_loss, = test_exe.run(fetch_list=[loss.name], feed=feed_dict)
- 相关 API :
 - :ref:`cn_api_fluid_ParallelExecutor`
 - :ref:`cn_api_fluid_BuildStrategy`
