.. _api_guide_sync_training_en:

####################################
Synchronous Distributed Training
####################################

Fluid supports data-parallel distributed synchronization training, the API uses the :code:`DistributedTranspiler` to convert a single machine network configuration into a :code:`pserver` side and :code:`trainer` side program that can be executed in multiple machines. The user executes the same piece of code on different nodes, depending on the environment variable or startup parameters.
You can execute the corresponding :code:`pserver` or :code:`trainer` role. Fluid distributed synchronization training supports both pserver mode and NCCL2 mode.There are differences in the use of the API, so you need to pay attention.

Pserver mode distributed training
======================================

API usage for details, please refer to :ref:`DistributeTranspiler`, simple example usage:

.. code-block:: python

	config = fluid.DistributedTranspilerConfig()
	#Configuring policy config
	config.slice_var_up = False
	t = fluid.DistributedTranspiler(config=config)
	t.transpile(trainer_id,
				program=main_program,
				pservers="192.168.0.1:6174,192.168.0.2:6174",
				trainers=1,
				sync_mode=True)

Among the above parameters:

- :code:`trainer_id` : The id of the trainer node, from 0 to n-1, where n is the number of trainer nodes in the current training task.
- :code:`program` : The converted :code:`program` is used :code:`fluid.default_main_program()` by default.
- :code:`pservers` : list of IP ports of the pserver node in the current training task
- :code:`trainers` : int type, the number of trainer nodes in the current training task. Please note:
	* In pserver mode, the number of trainer nodes can be different from the number of pserver nodes, such as 20 pservers and 50 trainers. In the actual training task, you can find the best performance by adjusting the number of pserver nodes and trainer nodes.
	* In NCCL2 mode, this parameter is a string specifying the IP port list of the trainer node.
- :code:`sync_mode` : Whether it is synchronous training mode, the default is True, this parameter is not passed also defaults to the synchronous training mode.


Among them, the supported config includes:

- :code:`slice_var_up` : Configure whether to split a parameter to multiple pservers for optimization, which is enabled by default. This option is applicable to scenes where the number of model parameters is small, but a large number of nodes are needed, which is beneficial to improve the computational parallelism of the pserver side.
- :code:`split_method` : Configure the transpiler allocates parameters (or slices of parameters) to multiple pservers. The default is "RoundRobin", also you can use "HashName".
- :code:`min_block_size` : If parameter splitting is configured, specify the minimum size of the Tensor to prevent the RPC request packet from being too small. The default is 8192, generally, you do not need to adjust this parameter.
- :code:`enable_dc_asgd` : Whether to enable :code:`DC-ASGD`. This option is effective in asynchronous training, using asynchronous training to compensate algorithm
- :code:`mode` : You can choose "pserver" or "nccl2" to specify distributed training using pserver mode or NCCL2 mode.
- :code:`print_log` : Whether to enable the transpiler debug log, this item is used for development and debugging

General environment variable configuration:

- :code:`FLAGS_rpc_send_thread_num` :int, specifies the number of threads when RPC communication is sent
- :code:`FLAGS_rpc_get_thread_num` : int, specifies the number of threads when RPC communication is accepted
- :code:`FLAGS_rpc_prefetch_thread_num` : int, the number of prefetch threads when the distributed lookup table executes RPC communication
- :code:`FLAGS_rpc_deadline` : int, the longest waiting time for RPC communication, in milliseconds, default 180000


NCCL2 mode distributed training
====================================

The multi-machine synchronous training mode based on NCCL2 (Collective Communication) is only supported under the GPU cluster.
Detailed API descriptions in this section can be found in :ref:`DistributeTranspiler` .

Note: In NCCL2 mode, the cluster does not need to start pserver, only need to start multiple trainer nodes.

Use the following code to convert the current :code:`Program` to Fluid :code:`Program` for NCCL2 distributed computing:

.. code-block:: python

	Config = fluid.DistributeTranspilerConfig()
	Config.mode = "nccl2"
	t = fluid.DistributedTranspiler(config=config)
	t.transpile(trainer_id,
				program=main_program,
				startup_program=startup_program,
				trainers="192.168.0.1:6174,192.168.0.2:6174",
				current_endpoint="192.168.0.1:6174")

among them:

- :code:`trainer_id` : The id of the trainer node, from 0 to n-1, where n is the number of trainer nodes in the current training task.
- :code:`program` and :code:`startup_program` : respectively for the main configuration program of the Fluid model and initializing the startup_program
- :code:`trainers` : String type, specifies the IP and port number of all trainers of the current task, only used for NCCL2 initialization (in pserver mode, this parameter is int, specifies the number of trainer nodes)
- :code:`current_endpoint` : the IP and port number of the current task's node.
