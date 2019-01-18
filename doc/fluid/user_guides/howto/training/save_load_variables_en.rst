.. _user_guide_save_load_vars:

######################################################
Save, load models or variables & incremental learning
######################################################

Model variable classification
##############################

In PaddlePaddle Fluid, all model variables are represented by :code:`fluid.Variable()` as the base class.Under this base class, model variables can be divided into the following categories:

1. Model parameter

  The model parameters are the variables trained and learned in the deep learning model. During the training process, the training framework calculates the current gradient of each model parameter according to the back propagation algorithm, and updates the parameters according to the gradient by the optimizer. The training process of the model can be seen as the process of continuously iterative updating of model parameters. In PaddlePaddle Fluid, the model parameters are represented by code:`fluid.framework.Parameter`, which is a derived class of :code:`fluid.Variable()` except: code:`fluid.Variable()` In addition to the various properties, :code:`fluid.framework.Parameter` can also be configured with its own initialization methods, update rate and other properties.

2. Long-term variable
  
  Long-term variables refer to variables that persist throughout the training process and are not destroyed by the end of an iteration, such as the global learning rate of dynamic adjustment. In PaddlePaddle Fluid, long-term variables are represented by setting the :code:`persistable` property of :code:`fluid.Variable()` to :code:`True`. All model parameters are long-term variables, but not all long-term variables are model parameters.

3. Temporary variables

  All model variables that do not belong to the above two categories are temporary variables. This type of variable exists only in one training iteration. After each iteration, all temporary variables are destroyed, and then before the next iteration, A new temporary variable will be constructed first for this iteration. In general, most of the variables in the model belong to this category, such as the input training data, the output of a normal layer, and so on.


How to save model variables
############################

The model variables we need to save are different depending on the application. For example, if we just want to save the model for future predictions, then just saving the model parameters is enough. But if we need to save a checkpoint for future recovery training, then we should save all the long-term variables, and even record the current epoch and step id. Because some model variables are not parameters, they are still essential for model training.

Save the model for prediction of new samples
=============================================

If we save the model for the purpose of predicting new samples, then just saving the model parameters is sufficient. We can use the :code:`fluid.io.save_params()` interface is used to save model parameters.

Example:

.. code-block:: python

    Import paddle.fluid as fluid

    Exe = fluid.Executor(fluid.CPUPlace())
    Param_path = "./my_paddle_model"
    Prog = fluid.default_main_program()
    Fluid.io.save_params(executor=exe, dirname=param_path, main_program=None)

In the above example, PaddlePaddle Fluid scans all model variables in the default:code:`fluid.Program`, ie:code:`prog`, by calling the :code:`fluid.io.save_params` function. All of the model parameters are saved and saved to the specified :code:`param_path`.



How to load model variables
#############################

Corresponding to the preservation of model variables, we provide two sets of APIs to load the parameters of the model and the long-term variables of the loaded model.

Load model for prediction of new samples
========================================

For models saved with :code:`fluid.io.save_params`, you can load them with :code:`fluid.io.load_params`.

Example:

.. code-block:: python

    import paddle.fluid as fluid

    exe = fluid.Executor(fluid.CPUPlace())
    param_path = "./my_paddle_model"
    prog = fluid.default_main_program()
    fluid.io.load_params(executor=exe, dirname=param_path,
                         main_program=prog)

In the above example, by calling the :code:`fluid.io.load_params` function, PaddlePaddle Fluid will scan all the model variables in :code:`prog`, filter out all the model parameters, and try to get from code: Read them in `param_path`.

It is important to note that the :code:`prog` must be exactly the same as the forward part of the :code:`prog` used when calling:code:`fluid.io.save_params` and cannot contain any parameter updates. Operation. If there is an inconsistency between the two, it may cause some variables not to be loaded correctly; if the parameter update operation is incorrectly included, it may cause the parameters to be changed during normal prediction. The relationship between these two:code:`fluid.Program` is similar to the relationship between training: code:`fluid.Program` and test:code:`fluid.Program`, see: :ref:`user_guide_test_while_training` .

Also, special care must be taken to run :code:`fluid.default_startup_program()` before calling :code:`fluid.io.load_params`. If you run later, it may overwrite the loaded model parameters and cause an error.



Prediction of the used models and parameters saving:
##################
The forecasting engine provides a storage prediction model: code:`fluid.io.save_inference_model` and the load prediction model :code:`fluid.io.load_inference_model`.

- :code:`fluid.io.save_inference_model`: Please refer to  :ref:`api_guide_inference` .
- :code:`fluid.io.load_inference_model`: Please refer to  :ref:`api_guide_inference` .



Incremental training
############
Incremental training means that a learning system can continuously learn new knowledge from new samples and preserve most of the knowledge that has been learned before. Therefore, incremental learning involves two points: saving the parameters that need to be persisted at the end of the last training, and loading the last saved persistent parameters at the beginning of the next training. Therefore incremental training involves the following APIs:
:code:`fluid.io.save_persistables`, :code:`fluid.io.load_persistables` .

Single machine incremental training
===========================
The general steps for incremental training for a single unit are as follows:

1. At the end of the training call :code:`fluid.io.save_persistables` Save the persistence parameter to the specified location.
2. After the training startup_program is executed successfully by the executor :code:`Executor`, call :code:`fluid.io.load_persistables` to load the previously saved persistence parameters.
3. Continue training with the executor :code:`Executor` or :code:`ParallelExecutor`.


Example:

.. code-block:: python

    import paddle.fluid as fluid

    Exe = fluid.Executor(fluid.CPUPlace())
    Path = "./models"
    Prog = fluid.default_main_program()
    Fluid.io.save_persistables(exe, path, prog)

In the above example, by calling the :code:`fluid.io.save_persistables` function, PaddlePaddle Fluid will find long-term variables from all model variables in the default: code:`fluid.Program`, which is: code:`prog`. Save them to the specified :code:`path` directory.


.. code-block:: python

    Import paddle.fluid as fluid

    Exe = fluid.Executor(fluid.CPUPlace())
    Path = "./models"
    Startup_prog = fluid.default_startup_program()
    Exe.run(startup_prog)
    Fluid.io.load_persistables(exe, path, startup_prog)
    Main_prog = fluid.default_main_program()
    Exe.run(main_prog)
    
In the above example, by calling the :code:`fluid.io.load_persistables` function, PaddlePaddle Fluid will find long-term variables from all model variables in the default: code:`fluid.Program`, which is: code:`prog`. Load them one by one from the specified :code:`path` directory and continue training.


The general steps for multi-machine incremental (without distributed large-scale sparse matrices) training are:
===========================
There are several differences between multi-machine incremental training and stand-alone incremental training:

1. At the end of the training call :code:`fluid.io.save_persistables` When saving the persistence parameters, it is not necessary for all trainers to call this method, usually the 0th trainer to save.
2. The parameters of multi-machine incremental training are loaded on the PServer side, and the trainer side does not need to load parameters. After the PServer is fully started, the trainer will synchronize the parameters from the PServer.

The general steps for multi-machine delta (do not enable distributed large-scale sparse matrices) training are:

1. Trainer 0 is called at the end of the training :code:`fluid.io.save_persistables` Save the persistence parameter to the specified :code:`path`.
2. Share all the parameters saved by trainer 0 to all PServers through HDFS, etc. (each PServer needs to have complete parameters).
3. After the training startup_program is successfully executed by the executor (:code:`Executor`), the PServer calls :code:`fluid.io.load_persistables` to load the persistence parameters saved by the 0th trainer.
4. The PServer continues to start PServer_program via the executor :code:`Executor`.
5. All training node trainers are trained normally through the executor :code:`Executor` or :code:`ParallelExecutor`.


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

In the above example, the 0 train is called by the :code:`fluid.io.save_persistables` function, PaddlePaddle Fluid will default from
:code:`fluid.Program` That is: find all long-term variables in all model variables of :code:`prog` and save them to the specified :code:`path` directory. The stored model is then uploaded to a location accessible to all PServers by invoking a third-party file system (such as HDFS).

For the PServer to be loaded with parameters during training, for example:


.. code-block:: bash
    hadoop fs -get /remote/$path $path


.. code-block:: python

    import paddle.fluid as fluid

    exe = fluid.Executor(fluid.CPUPlace())
    path = "./models"
	pserver_endpoints = "127.0.0.1:1001,127.0.0.1:1002"
	trainers = 4
	Training_role == "PSERVER"
	config = fluid.DistributeTranspilerConfig()
	t = fluid.DistributeTranspiler(config=config)
	t.transpile(trainer_id, pservers=pserver_endpoints, trainers=trainers, sync_mode=True)

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

In the above example, each PServer obtains the parameters saved by trainer 0 by calling the HDFS command, and obtains the PServer's :code:`fluid.Program` by configuration. PaddlePaddle Fluid will be from this :code:`fluid.Program` That is: find all long-term variables in all model variables of :code:`pserver_startup` and load them next through the specified :code:`path` directory.
