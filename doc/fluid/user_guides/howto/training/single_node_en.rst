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

After the configuration of model, the configurations of :code:`fluid.default_startup_program()` and :code:`fluid.default_main_program()` have been finished.

Initialize Parameters
#######################

Random Initialization of Parameters
====================================

After the configuration of model,the initialization of parameters will be written into :code:`fluid.default_startup_program()` . By running this program in :code:`fluid.Executor()` , the random initialization of parameters will be finished in global :code:`fluid.global_scope()` .For example:

.. code-block:: python

   exe = fluid.Executor(fluid.CUDAPlace(0))
   exe.run(program=fluid.default_startup_program())

Note that in multi-GPU training, the parameters should be initialized on GPU0 and then will be distributed to multiple graphic cards through :code:`fluid.ParallelExecutor` .


Load Predefined Parameters
===========================

In the neural network training, predefined models are usually loaded to continue training. For how to load predefined parameters, please refer to :ref:`user_guide_save_load_vars_en`.


Single-card Training
#####################

Single-card training can be performed through calling :code:`run()` of :code:`fluid.Executor()` to run training :code:`fluid.Program` .
In the runtime, feed data with :code:`run(feed=...)` and get persistable data with :code:`run(fetch=...)` . For example:

.. code-block:: python

   ...
   loss = fluid.layers.mean(...)

   exe = fluid.Executor(...)
   # the result is an numpy array
   result = exe.run(feed={"image": ..., "label": ...}, fetch_list=[loss])

Notes:

1. About data type supported by feed, please refer to the article :ref:`user_guide_feed_data_to_executor_en`.
2. The return value of :code:`Executor.run` is the variable value of :code:`fetch_list=[...]` .The fetched Variable must be persistable. :code:`fetch_list` can be fed with either Variable list or name list of variables . :code:`Executor.run` returns Fetch result list.
3. If the fetched data contain sequence information,  you can set :code:`exe.run(return_numpy=False, ...)` to directly get :code:`fluid.LoDTensor` . You can directly access the information in :code:`fluid.LoDTensor` .

Multi-card Training
#######################

In multi-card training, you can use :code:`fluid.ParallelExecutor` to run training :code:`fluid.Program`. For example:

.. code-block:: python

   train_exe = fluid.ParallelExecutor(use_cuda=True, loss_name=loss.name,
                                main_program=fluid.default_main_program())
   train_exe.run(fetch_list=[loss.name], feed={...})

Notes:

1. The constructor of :code:`ParallelExecutor` needs to be set with :code:`fluid.Program` to be run which can not be modified at runtime. The default value is :code:`fluid.default_main_program()` .
2. :code:`ParallelExecutor` should be indicated whether to use CUDA to train. In the mode of graphic card training, all graphic cards will be occupied. Users can configure `CUDA_VISIBLE_DEVICES <http://www.acceleware.com/blog/cudavisibledevices-masking-gpus>`_ to change graphics cards that are being used.

Advanced Usage
###############

.. toctree::
   :maxdepth: 2

   test_while_training_en.rst





