########
单机训练
########

准备工作
########

要进行PaddlePaddle Fluid单机训练，需要先 :ref:`user_guide_prepare_data` 和
:ref:`user_guide_configure_simple_model` 。当\
:ref:`user_guide_configure_simple_model` 完毕后，可以得到两个\
:code:`fluid.Program`， :code:`startup_program` 和 :code:`main_program`。
默认情况下，可以使用 :code:`fluid.default_startup_program()` 与\ :code:`fluid.default_main_program()` 获得全局的 :code:`fluid.Program`。

例如:

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

在上述模型配置执行完毕后， :code:`fluid.default_startup_program()` 与\
:code:`fluid.default_main_program()` 配置完毕了。

初始化参数
##########

参数随机初始化
==============

用户配置完模型后，参数初始化操作会被写入到\
:code:`fluid.default_startup_program()` 中。使用 :code:`fluid.Executor()` 运行
这一程序，初始化之后的参数默认被放在可在全局scope中，即 :code:`fluid.global_scope()` 。例如:

.. code-block:: python

   exe = fluid.Executor(fluid.CUDAPlace(0))
   exe.run(program=fluid.default_startup_program())

载入预定义参数
==============

在神经网络训练过程中，经常会需要载入预定义模型，进而继续进行训练。\
如何载入预定义参数，请参考 :ref:`user_guide_save_load_vars`。


单卡训练
########

执行单卡训练可以使用 :code:`fluid.Executor()` 中的 :code:`run()` 方法，运行训练\
:code:`fluid.Program` 即可。在运行的时候，用户可以通过 :code:`run(feed=...)`\
参数传入数据；用户可以通过 :code:`run(fetch=...)` 获取持久的数据。例如:\

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

多卡训练
#######################
在多卡训练中，你可以使用 :code:`fluid.compiler.CompiledProgram` 来编译 :code:`fluid.Program` ，然后调用 :code:`with_data_parallel` 。例如：

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

注释：

1. :ref:`cn_api_fluid_CompiledProgram` 会将传入的 :code:`fluid.Program` 转为计算图，即Graph，因为 :code:`compiled_prog` 与传入的 :code:`train_program` 是完全不同的对象，目前还不能够对 :code:`compiled_prog` 进行保存。
2. 多卡训练也可以使用 :ref:`cn_api_fluid_ParallelExecutor` ，但是现在推荐使用 :ref:`cn_api_fluid_CompiledProgram` .
3. 如果 :code:`exe` 是用CUDAPlace来初始化的，模型会在GPU中运行。在显卡训练模式中，所有的显卡都将被占用。用户可以配置 `CUDA_VISIBLE_DEVICES <http://www.acceleware.com/blog/cudavisibledevices-masking-gpus>`_ 以更改被占用的显卡。
4. 如果 :code:`exe` 是用CPUPlace来初始化的，模型会在CPU中运行。在这种情况下，多线程用于运行模型，同时线程的数目和逻辑核的数目相等。用户可以配置 ``CPU_NUM`` 以更改使用中的线程数目。

进阶使用
###############
.. toctree::
   :maxdepth: 2

   test_while_training.rst
