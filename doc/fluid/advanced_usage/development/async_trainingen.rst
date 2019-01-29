.. _api_guide_async_training_en:

############
Distributed asynchronous training
############

Fluid supports distributed asynchronous training for data parallelism. The API uses :code:`DistributedTranspiler` to convert a single network configuration into a :code:`pserver` side program and the :code:`trainer` side program that can be executed in multiple machines. The user executes the same piece of code on different nodes, depending on the environment variable or startup parameters, the corresponding :code:`pserver` or :code:`trainer` role can be executed. Fluid asynchronous training only supports the pserver mode. The main difference between asynchronous training and `synchronous training <../distributed/sync_training.html>`_ is that the gradient of each trainer is asynchronously updated to the parameters, while synchronous training is the uniform combination of all trainer's gradients and is updated to the parameters. Therefore, the hyperparameters of synchronous training and asynchronous training need to be adjusted separately.

Pserver mode distributed asynchronous training
=======================

API detailed usage please refer to :ref: `api_fluid_DistributeTranspiler` , simple example usage:

.. code-block:: python

	config = fluid.DistributedTranspilerConfig()
	#Configuring config policy 
	config.slice_var_up = False
	t = fluid.DistributedTranspiler(config=config)
	t.transpile(trainer_id,
				program=main_program,
				pservers="192.168.0.1:6174,192.168.0.2:6174",
				trainers=1,
				sync_mode=False)

For the above parameter description, please refer to `Sync Training <../distributed/sync_training.html>`_

Note that when doing asynchronous training, please modify the value of :code:`sync_mode`

- :code:`sync_mode` : Whether it is synchronous training mode, the default is True. If you do not pass this parameter, the default is synchronous training mode. If it is set to False, it is asynchronous training.
