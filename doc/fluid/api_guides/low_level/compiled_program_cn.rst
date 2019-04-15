..  _api_guide_compiled_program_zh:

################
CompiledProgram
################

:ref:`api_fluid_CompiledProgram` 用于把程序转化为不同的优化组合。例如，你可以使用with_data_parallel将程序转化为数据并行程序，使其能够运行在多个设备上。 


.. code-block:: python

    # 注释：
    #   - 如果你想在ParallelExecutor中指定用于运行的GPU卡，需要在环境中定义
    #     CUDA_VISIBLE_DEVICES
    #   - 如果你想在ParallelExecutor中使用多CPU来运行程序，需要在环境中定义
    #     CPU_NUM
    
    # 首先创建Executor。
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    # 运行启动程序仅一次。
    exe.run(fluid.default_startup_program())
    
    # 直接运行主程序，无需编译。
    loss = exe.run(fluid.default_main_program(),
                    feed=feed_dict,
                    fetch_list=[loss.name])
    
    # 或者编译程序后用数据并行方式运行模型。
    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = dev_count * 4 # the size of thread pool.
    build_strategy = fluid.BuildStrategy()
    build_strategy.memory_optimize = True if memory_opt else False  
    compiled_prog = compiler.CompiledProgram(
        fluid.default_main_program()).with_data_parallel(
            loss_name=loss.name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)
    loss, = exe.run(compiled_prog,
                    feed=feed_dict,
                    fetch_list=[loss.name])
                    
- 相关API :

 - :ref:`api_fluid_CompiledProgram`
