..  _api_guide_compiled_program:

################
CompiledProgram
################

:ref:`cn_api_fluid_CompiledProgram` 用于把程序转化为不同的优化组合。


.. code-block:: python

    # 首先创建 Executor。
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    # 运行启动程序仅一次。
    exe.run(fluid.default_startup_program())

    # 直接运行主程序，无需编译。
    loss = exe.run(fluid.default_main_program(),
                    feed=feed_dict,
                    fetch_list=[loss.name])

    # 或者编译程序后用优化过的 Program 运行模型。
    build_strategy = fluid.BuildStrategy()
    build_strategy.memory_optimize = True if memory_opt else False
    compiled_prog = compiler.CompiledProgram(
        fluid.default_main_program(),
        build_strategy=build_strategy)
    loss, = exe.run(compiled_prog,
                    feed=feed_dict,
                    fetch_list=[loss.name])

- 相关 API :

 - :ref:`cn_api_fluid_CompiledProgram`
