.. _api_guide_parallel_executor_en:

##############################
Parallel Executor
##############################

:code:`ParallelExecutor` is an upgraded version of Executor, in addition, it supports model training of :code:`Program` in parallel with data. Users can use the Python script to run :code:`ParallelExecutor`. The execution process of :code:`ParallelExecutor` is as follows:

- First it builds :code:`SSA Graph` and a thread pool based on :code:`Program`, the number of :code:`GPU` cards (or :code:`CPU` cores) and :ref:`api_fluid_BuildStrategy` ;
- During execution, it executes the Op depending on whether the input of Op is ready, so that multiple Ops that do not depend on each other can be executed in parallel in the thread pool;

When constructing :code:`ParallelExecutor`, you need to specify the device type of the current :code:`Program`, namely :code:`GPU` or :code:`CPU` :

* execution on :code:`GPU` : :code:`ParallelExecutor` will automatically detect the number of currently available :code:`GPU` s, and execute :code:`Program` on each :code:`GPU` . The user can also specify the :code:`GPU` that the executor can use by setting the :code:`CUDA_VISIBLE_DEVICES` environment variable;
* execution on multi-threaded :code:`CPU` : :code:`ParallelExecutor` will automatically detect the number of currently available :code:`CPU` s, and take it as the number of threads in the executor . Each thread executes :code:`Program` separately. The user can also specify the number of threads currently used for training by setting the :code:`CPU_NUM` environment variable.

:code:`ParallelExecutor` supports model training and model prediction:

* Model training: :code:`ParallelExecutor` aggregates the parameter gradients on multiple nodes during the execution process, and then updates the parameters;
* Model prediction: during the execution of :code:`ParallelExecutor`, each node runs the current :code:`Program` independently;

:code:`ParallelExecutor` supports two modes of gradient aggregation during model training, :code:`AllReduce` and :code:`Reduce` :

* In :code:`AllReduce` mode, :code:`ParallelExecutor` calls AllReduce operation to make the parameter gradients on multiple nodes completely equal, and then each node independently updates the parameters;
* In :code:`Reduce` mode, :code:`ParallelExecutor` will pre-allocate updates of all parameters to different nodes. During the execution :code:`ParallelExecutor` calls Reduce operation to aggregate parameter gradients on the pre-specified node, and the parameters are updated. Finally, the Broadcast operation is called to send the updated parameters to other nodes.

These two modes are specified by :code:`build_strategy`. For how to use them, please refer to :ref:`api_fluid_BuildStrategy` .

**Note**: If you use :code:`CPU` to execute :code:`Program` in multi-thread in Reduce mode, the parameters of :code:`Program` will be shared among multiple threads. On some models , Reduce mode can save a lot of memory.

Since the execution speed of the model is related to the model structure and the execution strategy of the executor, :code:`ParallelExecutor` allows you to modify the relevant parameters of the executor, such as the size of thread pool  ( :code:`num_threads` ), how many iterations should be done to clean up temporary variables :code:`num_iteration_per_drop_scope` . For more information, please refer to :ref:`api_fluid_ExecutionStrategy`.


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

    # Define train_exe and test_exe
    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = dev_count * 4 # the size of thread pool.
    build_strategy = fluid.BuildStrategy()
    build_strategy.memory_optimize = True if memory_opt else False

    train_exe = fluid.ParallelExecutor(use_cuda=use_cuda,
                                       main_program=train_program,
                                       build_strategy=build_strategy,
                                       exec_strategy=exec_strategy,
                                       loss_name=loss.name)
    # NOTE: loss_name is unnecessary for test_exe.
    test_exe = fluid.ParallelExecutor(use_cuda=True,
                                      main_program=test_program,
                                      build_strategy=build_strategy,
                                      exec_strategy=exec_strategy,
                                      share_vars_from=train_exe)

    train_loss, = train_exe.run(fetch_list=[loss.name], feed=feed_dict)
    test_loss, = test_exe.run(fetch_list=[loss.name], feed=feed_dict)

**Note**: :code:`fluid.Executor` and :code:`fluid.ParallelExecutor` are two completely different executors. First of all, their execution objects are different. The execution object of :code:`fluid.Executor` is :code:`fluid.Program` and the execution object of :code:`fluid.ParallelExecutor` is Graph. Secondly, their execution schedules are different. :code:`fluid.Executor` runs one by one according to the order of operators in :code:`fluid.Program`. :code:`fluid.ParallelExecutor` executes concurrently according to the dependencies between nodes in Graph.

- Related API :
 - :ref:`api_fluid_ParallelExecutor`
 - :ref:`api_fluid_BuildStrategy`
 - :ref:`api_fluid_ExecutionStrategy`
