.. _api_guide_parallel_executor_en:

##############################
Parallel Executor
##############################


:code:`ParallelExecutor` is an executor that executes :code:`Program` separately on multiple nodes in a data-parallelism manner. Users can use the Python script to run :code:`ParallelExecutor`. The execution process of :code:`ParallelExecutor` is as follows:

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

- Related API :
 - :ref:`api_fluid_ParallelExecutor`
 - :ref:`api_fluid_BuildStrategy`
 - :ref:`api_fluid_ExecutionStrategy`
