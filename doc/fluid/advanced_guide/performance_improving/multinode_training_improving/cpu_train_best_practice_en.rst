.. _api_guide_cpu_training_best_practice_en:

######################################################
Best practices of distributed training on CPU
######################################################

To improve the training speed of CPU distributed training, we must consider two aspects:

1. Improve the training speed mainly by improving utilization rate of CPU; 
2. Improve the communication speed mainly by reducing the amount of data transmitted in the communication.

Improve CPU utilization 
=============================

The CPU utilization mainly depends on :code:`ParallelExecutor`, which can make full use of the computing power of multiple CPUs to speed up the calculation.

For detailed API usage, please refer to :ref:`api_fluid_ParallelExecutor` . A simple example:

.. code-block:: python

	# Configure the execution strategy, mainly to set the number of threads
	exec_strategy = fluid.ExecutionStrategy()
	exec_strategy.num_threads = 8

	# Configure the composition strategy, for CPU training, you should use the Reduce mode for training.
	build_strategy = fluid.BuildStrategy()
	if int(os.getenv("CPU_NUM")) > 1:
		build_strategy.reduce_strategy=fluid.BuildStrategy.ReduceStrategy.Reduce

	pe = fluid.ParallelExecutor(
		use_cuda=False,
		loss_name=avg_cost.name,
		main_program=main_program,
		build_strategy=build_strategy,
		exec_strategy=exec_strategy)

Among the parameters above:

- :code:`num_threads` : the number of threads used by the model training. It is preferably close to the number of the physical CPU cores of the machine where the training is performed.
- :code:`reduce_strategy` : For CPU training, you should choose fluid.BuildStrategy.ReduceStrategy.Reduce


Configuration of general environment variables:

- :code:`CPU_NUM`: The number of replicas of the model, preferably the same as num_threads


Improve communication speed
==============================

To reduce the amount of communication data and improve communication speed is achieved mainly by using sparse updates, the current support for `sparse update <../layers/sparse_update_en.html>`_ is mainly :ref:`api_fluid_layers_embedding`.

.. code-block:: python

	data = fluid.layers.data(name='ids', shape=[1], dtype='int64')
	fc = fluid.layers.embedding(input=data, size=[dict_size, 16], is_sparse=True)

Among the parameters above:

- :code:`is_sparse`: Use sparse updates to configure embedding. If the dict_size of embedding is large but the number of data are very small each time, it is recommended to use the sparse update method.
