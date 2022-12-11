..  _api_guide_compiled_program_en:

################
CompiledProgram
################

The :ref:`api_fluid_CompiledProgram` is used to transform a program for various optimizations. For example, you can use :code:`with_data_parallel` to transform the program to data parallel program so that it can be run in multiple devices.


.. code-block:: python

    # Note:
    #   - If you want to specify the GPU cards which are used to run
    #     in ParallelExecutor, you should define the CUDA_VISIBLE_DEVICES
    #     in environment.
    #   - If you want to use multi CPU to run the program in ParallelExecutor,
    #     you should define the CPU_NUM in the environment.

    # First create the Executor.
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # Run the startup program once and only once.
    exe.run(fluid.default_startup_program())

    # Run the main program directly without compile.
    loss = exe.run(fluid.default_main_program(),
                    feed=feed_dict,
                    fetch_list=[loss.name])

    # Or, compiled the program, and then run the model with data parallel.
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

**Note**: :code:`fluid.Porgram` and :code:`compiler.CompiledPorgram` are completely different :code:`Programs`. :code:`fluid.Porgram` is composed of a series of operators. :code:`compiler.CompiledPorgram` compiles the :code:`fluid.Porgram` and converts it into a computational graph. :code:`compiler.CompiledPorgram` cannot be saved at present.


- Related API :
 - :ref:`api_fluid_CompiledProgram`
