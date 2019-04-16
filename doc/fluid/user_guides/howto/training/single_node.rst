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

在上述模型配置执行完毕后， :code:`fluid.default_startup_program()` 与\
:code:`fluid.default_main_program()` 配置完毕了。

初始化参数
##########

参数随机初始化
==============

用户配置完模型后，参数初始化操作会被写入到\
:code:`fluid.default_startup_program()` 中。使用 :code:`fluid.Executor()` 运行
这一程序，即可在全局 :code:`fluid.global_scope()` 中随机初始化参数。例如:

.. code-block:: python

   exe = fluid.Executor(fluid.CUDAPlace(0))
   exe.run(program=fluid.default_startup_program())

值得注意的是: 如果使用多GPU训练，参数需要先在GPU0上初始化，再经由\
:code:`fluid.ParallelExecutor` 分发到多张显卡上。


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

   ...
   loss = fluid.layers.mean(...)

多卡训练
#######################
在多卡训练中，你可以使用:code:`fluid.compiler.CompiledProgram`来编译:code:`fluid.Program`，然后调用:code:`with_data_parallel`。例如：

.. code-block:: python
   
    exe = fluid.Executor(...)
    
    compiled_prog = fluid.compiler.CompiledProgram(
        fluid.default_main_program()).with_data_parallel(
            loss_name=loss.name)
           
    result = exe.run(program=compiled_prog, 
                    fetch_list=[loss.name], 
                    feed={"image": ..., "label": ...}) 

注释：

1. :ref:`cn_api_fluid_CompiledProgram`的构造函数需要经过:code:`fluid.Program`设置后运行，这在运行时内无法被修改。
2. 如果:code:`exe`是用CUDAPlace来初始化的，模型会在GPU中运行。在显卡训练模式中，所有的显卡都将被占用。用户可以配置 `CUDA_VISIBLE_DEVICES <http://www.acceleware.com/blog/cudavisibledevices-masking-gpus>`_ 以更改被占用的显卡。
3. 如果:code:`exe`是用CPUPlace来初始化的，模型会在CPU中运行。在这种情况下，多线程用于运行模型，同时线程的数目和逻辑核的数目相等。用户可以配置`CPU_NUM`以更改使用中的线程数目。

进阶使用
###############
 @@ -98,8 +104,3 @@ Advanced Usage
   :maxdepth: 2

   test_while_training_en.rst
