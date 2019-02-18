.. _api_guide_parallel_executor_en:

##############################
Data Parallel Execution Engine
##############################


:code:`ParallelExecutor` is an executor that executes :code:`Program` separately on multiple nodes in a data-parallel manner. Users can use the Python script to execute :code:`ParallelExecutor`. The execution processes of:code:`ParallelExecutor` are as follows:

- First building :code:`SSA Graph` and a thread pool based on: :code:`Program`, the number of :code:`GPU` cards (or :code:`CPU` 's cores) and :ref:`api_fluid_BuildStrategy` ;
- During execution, execute the Op depending on whether the input of Op is ready, so that multiple Ops that do not depend on each other can be executed in parallel in the thread pool;

When constructing :code:`ParallelExecutor`, you need to specify the device type of the current :code:`Program`, :code:`GPU` or :code:`CPU` :

* Use :code:`GPU` to execution: :code:`ParallelExecutor` will automatically detect the current machine can use :code:`GPU` 's numbers, and execute :code:`Program` on each :code:`GPU`, the user can also specify the :code:`GPU` that the executor can use by setting the :code:`CUDA_VISIBLE_DEVICES` environment variable;
* Use :code:`CPU` multi-threaded execution: :code:`ParallelExecutor` will automatically detect the current machine: the number of the :code:`CPU` core, and takes it as the number of each thread in the executor , each thread executes :code:`Program` separately, the user can also specify the number of threads currently used for training by setting the :code:`CPU_NUM` environment variable.

:code:`ParallelExecutor` supports model training and model prediction:

* Model training: :code:`ParallelExecutor` aggregates the parameter gradients on multiple nodes during the execution process, and then update the parameters;
* Model prediction: during the execution of :code:`ParallelExecutor`, each node runs the current :code:`Program` independently;

:code:`ParallelExecutor` supports two modes of gradient aggregation during model training, :code:`AllReduce` and :code:`Reduce` :

* In :code:`AllReduce` mode, :code:`ParallelExecutor` calls AllReduce operation to make the parameter gradients on multiple nodes completely equal, and then each node independently updates the parameters;
* In :code:`Reduce` mode, :code:`ParallelExecutor` will pre-allocate all parameter updated to different nodes. During the execution :code:`ParallelExecutor` call Reduce operation to aggregation parameter gradients on the pre-specified node, and the parameters are updated. Finally, the Broadcast operation is called to send the updated parameters to other nodes.

These two modes are specified by :code:`build_strategy`. For how to use them, please refer to :ref:`api_fluid_BuildStrategy` .

**Note**: If you use :code:`CPU` to execute :code:`Program` multithreadedly in Reduce mode, the parametersof :code:`Program`  are shared among multiple threads. On some models , Reduce mode can save a lot of memory.

Since the execution speed of the model is related to the model structure and the execution strategy of the executor, :code:`ParallelExecutor` allows the user to modify the relevant parameters of the executor, such as: thread pool size ( :code:`num_threads` ), how many iterations are cleaned up a temporary variable: code:`num_iteration_per_drop_scope`, etc. For more information, please refer to :ref:`api_fluid_ExecutionStrategy`.

- Related API summary:
 - :ref:`api_fluid_ParallelExecutor`
 - :ref:`api_fluid_BuildStrategy`
 - :ref:`api_fluid_ExecutionStrategy`
