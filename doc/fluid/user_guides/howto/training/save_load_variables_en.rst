.. _user_guide_save_load_vars_en:

##########################################################
Save, Load and Incremental Learning of Models or Variables
##########################################################

Model variable classification
##############################

In PaddlePaddle Fluid, all model variables are represented by :code:`fluid.Variable()` as the base class. Under this base class, model variables can be divided into the following categories:

1. Model parameter

  The model parameters are the variables trained and learned in the deep learning model. During the training process, the training framework calculates the current gradient of each model parameter according to the back propagation algorithm, and updates the parameters according to their gradients by the optimizer. The essence of the training process of a model can be seen as the process of continuously iterative updating of model parameters. In PaddlePaddle Fluid, the model parameters are represented by :code:`fluid.framework.Parameter` , which is a derived class of :code:`fluid.Variable()` . Besides various properties of :code:`fluid.Variable()` , :code:`fluid.framework.Parameter` can also be configured with its own initialization methods, update rate and other properties.

2. Persistable variable
  
  Persistable variables refer to variables that persist throughout the training process and are not destroyed by the end of an iteration, such as the global learning rate which is dynamically adjusted. In PaddlePaddle Fluid, persistable variables are represented by setting the :code:`persistable` property of :code:`fluid.Variable()` to :code:`True`. All model parameters are persistable variables, but not all persistable variables are model parameters.

3. Temporary variables

  All model variables that do not belong to the above two categories are temporary variables. This type of variable exists only in one training iteration. After each iteration, all temporary variables will be destroyed, and before the next iteration, A new set of temporary variables will be constructed first for this iteration. In general, most of the variables in the model belong to this category, such as the input training data, the output of a normal layer, and so on.


How to save model variables
############################

The model variables we need to save are different depending on the application. For example, if we just want to save the model for future predictions, just saving the model parameters will be enough. But if we need to save a checkpoint for future recovery of current training, then we should save all the persistable variables, and even record the current epoch and step id. It is because even though some model variables are not parameters, they are still essential for model training.

differences among save_vars、save_params、save_persistables and save_inference_model
###################################################################################
 1. :code:`save_inference_model` will trim the network according to :code:`feeded_var_names` and :code:`target_vars` configured by users, and save the ``__model__`` of network structures and long-term variables of networks after the trim.

  2. :code:`save_persistables` will not save network structures but will save all the long-term variables in networks in the appointed location.

  3. :code:`save_params` will not save network structures but will save all the model parameters in networks in the appointed location.

  4. :code:`save_vars` will not save network structures but will save according to  :code:`fluid.framework.Parameter` list appointed by users.

   :code:`save_persistables` can save most comprehensive network parameters. In incremental training or recovery training situation, please choose :code:`save_persistables` to save variables.
  :code:`save_inference_model` will save network parameters and models after trim. For later inference, please choose  :code:`save_inference_model` to save variables and networks.
  :code:`save_vars 和 save_params` is only used in the situation where users know clearly about the uses or for special purpose, and is not recommended in general.


Save the model to make prediction for new samples
===================================================

If we save the model to make prediction for new samples, just saving the model parameters will be sufficient. We can use the :code:`fluid.io.save_params()` interface to save model parameters.

For example:

.. code-block:: python

    import paddle.fluid as fluid

    exe = fluid.Executor(fluid.CPUPlace())
    param_path = "./my_paddle_model"
    prog = fluid.default_main_program()
    fluid.io.save_params(executor=exe, dirname=param_path, main_program=None)

In the example above, by calling the :code:`fluid.io.save_params` function, PaddlePaddle Fluid scans all model variables in the default :code:`fluid.Program` , i.e. :code:`prog` and picks out all model parameters. All these model parameters are saved to the specified :code:`param_path` .



How to load model variables
#############################

Corresponding to saving of model variables, we provide two sets of APIs to load the model parameters and the persistable variables of model.

Load model to make predictions for new samples
================================================

For models saved with :code:`fluid.io.save_params` , you can load them with :code:`fluid.io.load_params`.

For example:

.. code-block:: python

    import paddle.fluid as fluid

    exe = fluid.Executor(fluid.CPUPlace())
    param_path = "./my_paddle_model"
    prog = fluid.default_main_program()
    fluid.io.load_params(executor=exe, dirname=param_path,
                         main_program=prog)

In the above example, by calling the :code:`fluid.io.load_params` function, PaddlePaddle Fluid will scan all the model variables in :code:`prog`, filter out all the model parameters, and try to load them from :code:`param_path` .

It is important to note that the :code:`prog` here must be exactly the same as the forward part of the :code:`prog` used when calling :code:`fluid.io.save_params` and cannot contain any operations of parameter updates. If there is an inconsistency between the two, it may cause some variables not to be loaded correctly; if the parameter update operation is incorrectly included, it may cause the parameters to be changed during normal prediction. The relationship between these two :code:`fluid.Program` is similar to the relationship between training :code:`fluid.Program` and test :code:`fluid.Program`, see: :ref:`user_guide_test_while_training_en` .

In addition, special care must be taken that :code:`fluid.default_startup_program()` **must** be run before calling :code:`fluid.io.load_params` . If you run it later, it may overwrite the loaded model parameters and cause an error.



Prediction of the used models and parameters saving
#######################################################


The inference engine provides two interfaces : prediction model saving :code:`fluid.io.save_inference_model` and the prediction model loading :code:`fluid.io.load_inference_model`.

- :code:`fluid.io.save_inference_model`: Please refer to  :ref:`api_guide_inference` .
- :code:`fluid.io.load_inference_model`: Please refer to  :ref:`api_guide_inference` .



Incremental training
#####################

Incremental training means that a learning system can continuously learn new knowledge from new samples and preserve most of the knowledge that has been learned before. Therefore, incremental learning involves two points: saving the parameters that need to be persisted at the end of the last training, and loading the last saved persistent parameters at the beginning of the next training. Therefore incremental training involves the following APIs:
:code:`fluid.io.save_persistables`, :code:`fluid.io.load_persistables` .

Single-node incremental training
=================================

The general steps of incremental training on a single unit are as follows:

1. At the end of the training, call :code:`fluid.io.save_persistables` to save the persistable parameter to the specified location.
2. After the training startup_program is executed successfully by the executor :code:`Executor`, call :code:`fluid.io.load_persistables` to load the previously saved persistable parameters.
3. Continue training with the executor :code:`Executor` or :code:`ParallelExecutor`.


Example:

.. code-block:: python

    import paddle.fluid as fluid

    exe = fluid.Executor(fluid.CPUPlace())
    path = "./models"
    prog = fluid.default_main_program()
    fluid.io.save_persistables(exe, path, prog)

In the above example, by calling the :code:`fluid.io.save_persistables` function, PaddlePaddle Fluid will find all persistable variables from all model variables in the default :code:`fluid.Program`, e.t. :code:`prog` , and save them to the specified :code:`path` directory.


.. code-block:: python

    import paddle.fluid as fluid

    exe = fluid.Executor(fluid.CPUPlace())
    path = "./models"
    startup_prog = fluid.default_startup_program()
    exe.run(startup_prog)
    fluid.io.load_persistables(exe, path, startup_prog)
    main_prog = fluid.default_main_program()
    exe.run(main_prog)
    
In the above example, by calling the :code:`fluid.io.load_persistables` function, PaddlePaddle Fluid will find persistable variables from all model variables in the default :code:`fluid.Program` , e.t. :code:`prog` . and load them one by one from the specified :code:`path` directory to continue training.


The general steps for multi-node incremental training (without distributed large-scale sparse matrices)
=========================================================================================================

There are several differences between multi-node incremental training and single-node incremental training:

1. At the end of the training, when :code:`fluid.io.save_persistables` is called to save the persistence parameters, it is not necessary for all trainers to call this method, usually it is called on the 0th trainer.
2. The parameters of multi-node incremental training are loaded on the PServer side, and the trainer side does not need to load parameters. After the PServers are fully started, the trainer will synchronize the parameters from the PServer.
3. In the situation where increment needs to be used determinately, multi-node needs to appoint ``current_endpoint`` parameter when calling :code:`fluid.DistributeTranspiler.transpile` .

The general steps for multi-node incremental training (do not enable distributed large-scale sparse matrices) are:

1. At the end of the training, Trainer 0 will call :code:`fluid.io.save_persistables` to save the persistable parameters to the specified :code:`path`.
2. Share all the parameters saved by trainer 0 to all PServers through HDFS or other methods. (each PServer needs to have complete parameters).
3. After the training startup_program is successfully executed by the executor ( :code:`Executor` ), the PServer calls :code:`fluid.io.load_persistables` to load the persistable parameters saved by the 0th trainer.
4. The PServer continues to start PServer_program via the executor :code:`Executor`.
5. All training node trainers conduct training process normally through the executor :code:`Executor` or :code:`ParallelExecutor` .


For trainers whose parameters are to be saved during training, for example:

.. code-block:: python

    import paddle.fluid as fluid

    exe = fluid.Executor(fluid.CPUPlace())
    path = "./models"
    trainer_id = 0
    if trainer_id == 0:
        prog = fluid.default_main_program()
        fluid.io.save_persistables(exe, path, prog)


.. code-block:: bash
    hadoop fs -mkdir /remote/$path
    hadoop fs -put $path /remote/$path

In the above example, the 0 trainer calls the :code:`fluid.io.save_persistables` function. By calling this function,  PaddlePaddle Fluid will find all persistable variables in all model variables from default :code:`fluid.Program` , e.t.  :code:`prog` , and save them to the specified :code:`path` directory. The stored model is then uploaded to a location accessible for all PServers by invoking a third-party file system (such as HDFS).

For the PServer to be loaded with parameters during training, for example:


.. code-block:: python

    import paddle.fluid as fluid

    exe = fluid.Executor(fluid.CPUPlace())
    path = "./models"
	pserver_endpoints = "127.0.0.1:1001,127.0.0.1:1002"
	trainers = 4
	Training_role == "PSERVER"
	config = fluid.DistributeTranspilerConfig()
	t = fluid.DistributeTranspiler(config=config)
	t.transpile(trainer_id, pservers=pserver_endpoints, trainers=trainers, sync_mode=True, current_endpoint=current_endpoint)

	if training_role == "PSERVER":
		current_endpoint = "127.0.0.1:1001"
		pserver_prog = t.get_pserver_program(current_endpoint)
		pserver_startup = t.get_startup_program(current_endpoint, pserver_prog)

		exe.run(pserver_startup)
		fluid.io.load_persistables(exe, path, pserver_startup)
		exe.run(pserver_prog)
	if training_role == "TRAINER":
		main_program = t.get_trainer_program()
				exe.run(main_program)

In the above example, each PServer obtains the parameters saved by trainer 0 by calling the HDFS command, and obtains the PServer's :code:`fluid.Program` by configuration. PaddlePaddle Fluid will find all persistable variables in all model variables from this :code:`fluid.Program` , e.t. :code:`pserver_startup` , and load them from the specified :code:`path` directory.
