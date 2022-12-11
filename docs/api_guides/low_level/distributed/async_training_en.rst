.. _api_guide_async_training_en:

####################################
Asynchronous Distributed Training
####################################

Fluid supports parallelism asynchronous distributed training. :code:`DistributeTranspiler` converts a single node network configuration into a :code:`pserver` side program and the :code:`trainer` side program that can be executed on multiple machines. The user executes the same piece of code on different nodes. Depending on the environment variables or startup parameters, the corresponding :code:`pserver` or :code:`trainer` role can be executed.

**Asynchronous distributed training in Fluid only supports the pserver mode** . The main difference between asynchronous training and `synchronous training <../distributed/sync_training_en.html>`_ is that the gradients of each trainer are asynchronously applied on the parameters, but in synchronous training, the gradients of all trainers must be combined first and then they are used to update the parameters. Therefore, the hyperparameters of synchronous training and asynchronous training need to be adjusted separately.

Asynchronous distributed training in Pserver mode
==================================================

For detailed API, please refer to :ref:`api_fluid_transpiler_DistributeTranspiler` . A simple example:

.. code-block:: python

    config = fluid.DistributeTranspilerConfig()
    #Configuring config policy
    config.slice_var_up = False
    t = fluid.DistributeTranspiler(config=config)
    t.transpile(trainer_id,
                program=main_program,
                pservers="192.168.0.1:6174,192.168.0.2:6174",
                trainers=1,
                sync_mode=False)

For the description of parameters above, please refer to `Sync Training <../distributed/sync_training_en.html>`_ .

Note that when performing asynchronous training, please modify the value of :code:`sync_mode` .

- :code:`sync_mode` : Whether it is synchronous training mode, the default is True. If you do not pass this parameter, the default is synchronous training mode. If it is set to False, it is asynchronous training.
