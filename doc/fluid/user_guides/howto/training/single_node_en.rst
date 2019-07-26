#####################
Single-node training
#####################

Preparation
############

To perform single-node training in PaddlePaddle Fluid, you need to read :ref:`user_guide_prepare_data_en` and :ref:`user_guide_configure_simple_model_en` . When you have finished reading :ref:`user_guide_configure_simple_model_en` , you can get two :code:`fluid.Program`, namely :code:`startup_program` and :code:`main_program` . By default, you can use :code:`fluid.default_startup_program()` and :code:`fluid.default_main_program()` to get global :code:`fluid.Program` .

For example:

.. code-block:: python

   import paddle.fluid as fluid

   image = fluid.layers.data(name="image", shape=[784])
   label = fluid.layers.data(name="label", shape=[1])
   hidden = fluid.layers.fc(input=image, size=100, act='relu')
   prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
   loss = fluid.layers.cross_entropy(input=prediction, label=label)
   loss = fluid.layers.mean(loss)

   sgd = fluid.optimizer.SGD(learning_rate=0.001)
   sgd.minimize(loss)

   # Here the fluid.default_startup_program() and fluid.default_main_program()
   # has been constructed.

After the configuration of model, the configurations of :code:`fluid.default_startup_program()` and :code:`fluid.default_main_program()` have been finished.

Initialize Parameters
#######################

Random Initialization of Parameters
====================================

After the configuration of model,the initialization of parameters will be written into :code:`fluid.default_startup_program()` . By running this program in :code:`fluid.Executor()` , the random initialization of parameters will be finished in global scope, i.e. :code:`fluid.global_scope()` .For example:

.. code-block:: python

   exe = fluid.Executor(fluid.CUDAPlace(0))
   exe.run(program=fluid.default_startup_program())

Load Predefined Parameters
===========================

In the neural network training, predefined models are usually loaded to continue training. For how to load predefined parameters, please refer to :ref:`user_guide_save_load_vars_en`.


Single-card Training
#####################

Single-card training can be performed through calling :code:`run()` of :code:`fluid.Executor()` to run training :code:`fluid.Program` .
In the runtime, feed data with :code:`run(feed=...)` and get persistable data with :code:`run(fetch=...)` . For example:

.. code-block:: python

    import paddle.fluid as fluid
    import numpy

    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(train_program, startup_program):
        data = fluid.layers.data(name='X', shape=[1], dtype='float32')
        hidden = fluid.layers.fc(input=data, size=10)
        loss = fluid.layers.mean(hidden)
        sgd = fluid.optimizer.SGD(learning_rate=0.001)
        sgd.minimize(loss)

    use_cuda = True
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # Run the startup program once and only once.
    # Not need to optimize/compile the startup program.
    startup_program.random_seed=1
    exe.run(startup_program)

    # Run the main program directly without compile.
    x = numpy.random.random(size=(10, 1)).astype('float32')
    loss_data, = exe.run(train_program,
                         feed={"X": x},
                         fetch_list=[loss.name])
    # Or 
    # compiled_prog = compiler.CompiledProgram(train_program)
    # loss_data, = exe.run(compiled_prog,
    #              feed={"X": x},
    #              fetch_list=[loss.name])

Notes:

1. About data type supported by feed, please refer to the article :ref:`user_guide_feed_data_to_executor_en`.
2. The return value of :code:`Executor.run` is the variable value of :code:`fetch_list=[...]` .The fetched Variable must be persistable. :code:`fetch_list` can be fed with either Variable list or name list of variables . :code:`Executor.run` returns Fetch result list.
3. If the fetched data contain sequence information,  you can set :code:`exe.run(return_numpy=False, ...)` to directly get :code:`fluid.LoDTensor` . You can directly access the information in :code:`fluid.LoDTensor` .

Multi-card Training
#######################
In multi-card training, you can use :code:`fluid.compiler.CompiledProgram` to compile the :code:`fluid.Program`, and then call :code:`with_data_parallel`. For example:

.. code-block:: python

    # NOTE: If you use CPU to run the program, you need
    # to specify the CPU_NUM, otherwise, fluid will use
    # all the number of the logic core as the CPU_NUM,
    # in that case, the batch size of the input should be
    # greater than CPU_NUM, if not, the process will be
    # failed by an exception.
    if not use_cuda:
        os.environ['CPU_NUM'] = str(2)

    compiled_prog = compiler.CompiledProgram(
        train_program).with_data_parallel(
        loss_name=loss.name)
    loss_data, = exe.run(compiled_prog,
                         feed={"X": x},
                         fetch_list=[loss.name])

Notes:

1. :ref:`api_fluid_CompiledProgram` will convert the input Program into a computational graph, and :code:`compiled_prog` is a completely different object from the incoming :code:`train_program`. At present, :code:`compiled_prog` can not be saved.
2. Multi-card training can also be used: ref:`api_fluid_ParallelExecutor` , but now it is recommended to use: :ref:`api_fluid_CompiledProgram`.
3. If :code:`exe` is initialized with CUDAPlace, the model will be run in GPU. In the mode of graphics card training, all graphics card will be occupied. Users can configure `CUDA_VISIBLE_DEVICES <http://www.acceleware.com/blog/cudavisibledevices-masking-gpus>`_ to change graphics cards that are being used. 
4. If :code:`exe` is initialized with CPUPlace, the model will be run in CPU. In this situation, the multi-threads are used to run the model, and the number of threads is equal to the number of logic cores. Users can configure `CPU_NUM`  to change the number of threads that are being used. 

Advanced Usage
###############

.. toctree::
   :maxdepth: 2

   test_while_training_en.rst
