########
Single-node training
########

Preparation
############

To perform single-node training in PaddlePaddle Fluid, you need to :ref:`user_guide_prepare_data` and :ref:`user_guide_configure_simple_model` . When :ref:`user_guide_configure_simple_model` is finished,you can get two :code:`fluid.Program`, :code:`startup_program` and :code:`main_program`.By default,you can use :code:`fluid.default_startup_program()` and :code:`fluid.default_main_program()` to get global :code:`fluid.Program` .

For example:

.. code-block:: python

   import paddle.fluid as fluid

   image = fluid.layers.data(name="image", shape=[784])
   label = fluid.layers.data(name="label", shape=[1])
   hidden = fluid.layers.fc(input=image, size=100, act='relu')
   prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
   loss = fluid.layers.mean(
       fluid.layers.cross_entropy(
           input=prediction,
           label=label
       )
   )

   sgd = fluid.optimizer.SGD(learning_rate=0.001)
   sgd.minimize(loss)

   # Here the fluid.default_startup_program() and fluid.default_main_program()
   # has been constructed.

After the configuration of model,the configurations of :code:`fluid.default_startup_program()` and :code:`fluid.default_main_program()` have been finished.

Initialized Parameters
#######################

Random Initialization of Parameters
====================================

After the configuration of model,the initialization of parameters will be written into :code:`fluid.default_startup_program()` .The random initialization of parameters will be finished in global with :code:`fluid.Executor()` .For example:

.. code-block:: python

   exe = fluid.Executor(fluid.CUDAPlace(0))
   exe.run(program=fluid.default_startup_program())

Attention needs to be paid that parameters should be initialized in GPU0 and then submitted to graphics cards through :code:`fluid.ParallelExecutor` when you use GPUs to train.


Load Predefined Parameters
===========================

During the neural network train, you often need to load predefined model to continue your train.About how to predefine parameters,please refer to :ref:`user_guide_save_load_vars`.


Single-card Training
#####################

You can use :code:`run()` of :code:`fluid.Executor()` to perform single-card train.What you need to do is run :code:`fluid.Program` .At the runtime,user can realize data feeding with :code:`run(feed=...)` and get prolong data with :code:`run(fetch=...)` .For example:\

.. code-block:: python

   ...
   loss = fluid.layers.mean(...)

   exe = fluid.Executor(...)
   # the result is an numpy array
   result = exe.run(feed={"image": ..., "label": ...}, fetch_list=[loss])

Notes:

1. About data format of feed,please refer to the article :ref:`user_guide_feed_data_to_executor`.
2. Return value of :code:`Executor.run` is the variable value of :code:`fetch_list=[...]` .The fetched Variable must be persistable. :code:`fetch_list` can be feed to either Variable list or name list of Variable. :code:`Executor.run` return Fetch result list.
3. If you need to fetch data with sequence information,you set :code:`exe.run(return_numpy=False, ...)` to directly get :code:`fluid.LoDTensor`. User can directly visit the information of :code:`fluid.LoDTensor` .

Multiple-card Training
#######################

In multiple-card training,you can use :code:`fluid.ParallelExecutor` to run :code:`fluid.Program`.For example:

.. code-block:: python

   train_exe = fluid.ParallelExecutor(use_cuda=True, loss_name=loss.name,
                                main_program=fluid.default_main_program())
   train_exe.run(fetch_list=[loss.name], feed={...})

Notes:

1. The constructed function of :code:`ParallelExecutor` needs to be indicated  :code:`fluid.Program` which can not be modified at runtime. And the default value is :code:`fluid.default_main_program()` .
2. :code:`ParallelExecutor` should be indicated whether to use CUDA to train. In the mode of graphics card training, all graphics cards will be unavailable.Users can configure `CUDA_VISIBLE_DEVICES <http://www.acceleware.com/blog/cudavisibledevices-masking-gpus>`_ to adjust graphics cards that are being used.

Advanced Usage
###############

.. toctree::
   :maxdepth: 2

   test_while_training
