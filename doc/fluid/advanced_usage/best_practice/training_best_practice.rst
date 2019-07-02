.. training_best_practice:

#####################
单机训练最佳实践
#####################

开始优化您的单机训练任务
-------------------------

PaddlePaddle Fluid可以支持在现代CPU、GPU平台上进行训练。如果您发现Fluid在进行单机训练的速度较慢，您可以根据这篇文档的建议对您的Fluid程序进行优化。

神经网络训练代码通常由三个步骤组成：网络构建、数据准备、模型训练。这篇文档将分别从这三个方向介绍Fluid训练中常用的优化方法。


1. 网络构建过程中的配置优化
=============

这部分优化与具体的模型有关，在这里，我们列举出一些优化过程中遇到过的一些示例。

1.1 cuDNN操作的选择
^^^^^^^^^^^^^^^^

`cuDNN <https://github.com/NVIDIA/nccl>`_ 是NVIDIA提供的深度神经网络计算库，其中包含了很多神经网络中常用算子，Paddle中的部分Op底层调用的是cuDNN库，例如 :code:`conv2d` ：

.. code-block:: python

    paddle.fluid.layers.conv2d(input,
                               num_filters,
                               filter_size,
                               stride=1,
                               padding=0,
                               dilation=1,
                               groups=None,
                               param_attr=None,
                               bias_attr=None,
                               use_cudnn=True,
                               act=None,
                               name=None)

在 :code:`use_cudnn=True` 时，框架底层调用的是cuDNN中的卷积操作。

通常cuDNN库提供的操作具体很好的性能表现，其性能明显优于Paddle原生的CUDA实现，比如 :code:`conv2d` 。但是cuDNN中有些操作的性能较差，比如： :code:`conv2d_transpose` 在 :code:`batch_size=1` 时、:code:`pool2d` 在 :code:`global_pooling=True` 时等，这些情况下，cuDNN实现的性能差于Paddle的CUDA实现，建议手动设置 :code:`use_cudnn=False` 。

1.2 使用融合功能的API
^^^^^^^^^^^^^^^^

Paddle提供一些粗粒度的API，这些API融合了多个细粒度API的计算，比如：

.. code-block:: python

    logits = fluid.layers.softmax(logits)
    loss = fluid.layers.cross_entropy(logits, label, ignore_index=255)

和

.. code-block:: python

    loss = fluid.layers.softmax_with_cross_entropy(logits, label, ignore_index=255, numeric_stable_mode=True)

用户网络配置中使用融合功能的API，通常能取得更好的计算性能。

2. 数据准备优化
=============

2.1 分析数据准备部分的耗时
^^^^^^^^^^^^^^^^

数据准备部分通常分为两个部分：数据读取部分和预处理部分。

- 数据读取部分：用户需要在Python端从磁盘中加载数据，然后将数据feed到Fluid的执行器中。
- 数据预处理部分：用户需要在Python端进行数据预处理，比如图像任务通常需要进行数据增强、裁剪等。

Fluid提供了两种数据读取方式：**同步数据读取** 和 **异步数据读取**，详情请参考文档 `如何准备数据 <http://paddlepaddle.org/documentation/docs/zh/1.5/user_guides/howto/prepare_data/index_cn.html>`_ 。


2.1.1 同步数据读取
>>>>>>>>>>>>>>>

同步数据读取是一种简单并且直观的数据准备方式，代码示例如下：

.. code-block:: python

    # 读取数据
    end = time.time()
    for batch_id, batch in enumerate(train_reader):
        data_time = time.time() - end
        # 训练网络
        executor.run(feed=[...], fetch_list=[...])
        batch_time = time.time() - end
        end = time.time()

用户通过调用自己编写的reader函数，reader每次输出一个batch的数据，并将数据传递给执行器。因此数据准备和执行是顺序进行的，用户可通过加入Python计时函数 :code`time.time()` 来统计数据准备部分和执行部分所占用的时间。

2.1.2 异步数据读取
>>>>>>>>>>>>>>>

Paddle里面使用 :code`py_reader` 接口来实现异步数据读取，代码示例如下：

.. code-block:: python

    # 启动py_reader
    train_py_reader.start()
    batch_id = 0
    try:
        end = time.time()
        while True:
            print("queue size: ", train_py_reader.queue.size())
            loss, = executor.run(fetch_list=[...])
            # ...
            batch_time = time.time() - end
            end = time.time()
            batch_id += 1
    except fluid.core.EOFException:
        train_py_reader.reset()

使用异步数据读取时，Paddle的C++端会维护一个数据队列，Python端通过单独的线程向C++端的数据队列传入数据。用户可以在训练过程中输出数据队列中数据的个数，如果queue size始终不为空，表明Python端数据准备的速度比模型执行的速度快，这种情况下Python端的数据读取可能不是瓶颈。

此外，Paddle提供的一些FLAGS也能很好的帮助分析性能，比如通过设置 :code:`export FLAGS_reader_queue_speed_test_mode=True` ，数据队列中的训练数据在被读取之后，不会从数据队列中弹出，这样能够保证数据队列始终不为空，这样就能够很好的评估出数据读取所占的开销。**注意，FLAGS_reader_queue_speed_test_mode只能在分析的时候打开，正常训练模型时需要关闭**。

2.2 优化数据准备速度的方法
^^^^^^^^^^^^^^^^

- 为降低训练的整体时间，建议用户使用异步数据读取的方式，并开启use_double_buffer。此外，用户可根据模型的实际情况设置数据队列的大小。
- 如果数据准备的时间大于模型执行的时间，或者出现了数据队列为空的情况，这时候需要考虑对Python的用户reader进行加速。常用的方法为：**使用Python多进程准备数据**。一个简单的使用多进程准备数据的示例，请参考 `YOLOv3 <https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/yolov3/reader.py>`_ 。
- Python端的数据预处理，都是使用CPU完成。如果Paddle提供了相应功能的API，可将这部分预处理功能写到模型配置中，如此Paddle就可以使用GPU来完成该预处理功能，这样也可以减轻CPU预处理数据的负担，提升总体训练速度。

3. 模型训练相关优化
=============

3.1 执行器介绍
^^^^^^^^^^^^^^^^

目前Paddle中有两个执行器， :code:`Executor` 和 :code:`ParallelExecutor` ，这两个执行器的区别：

执行调度器
>>>>>>>>>>>>>>>

..  csv-table:: 
    :header: "执行器 ", "执行对象", "执行策略"
    :widths: 3, 3, 5

    ":code:`Executor`",         ":code:`Program`",   "根据 :code:`Program` 中Operator定义的先后顺序依次运行。"
    ":code:`ParallelExecutor`", "SSA Graph", "根据Graph中各个节点之间的依赖关系，通过多线程运行。"

为了更好的分析模型， :code:`ParallelExecutor` 内部首先会将输入的 :code:`Program` 转为SSA Graph，然后根据 :code:`build_strategy` 中的配置，通过一系列的Pass对Graph进行优化，比如：memory optimize，operator fuse等优化。最后根据 :code:`execution_strategy` 中的配置执行训练任务。

此外， :code:`ParallelExecutor` 支持支持数据并行，即单进程多卡和多进程多卡，关于 :code:`ParallelExecutor` 的具体介绍请参考 `文档 <http://www.paddlepaddle.org/documentation/docs/en/1.5/api_guides/low_level/parallel_executor_en.html>`_ .

为了统一 :code:`ParallelExecutor` 接口和 :code:`Executor` 接口，Paddle提供了 :code:`fluid.compiler.CompiledProgram` 接口，在数据并行模式下，该接口底层调用的是 :code:`ParallelExecutor` 。

3.2 BuildStrategy中参数配置说明
^^^^^^^^^^^^^^^^
BuildStrategy配置选项
>>>>>>>>>>>>>>>

..  csv-table:: 
    :header: "选项", "类型", "默认值", "说明"
    :widths: 3, 3, 3, 5

    ":code:`reduce_strategy`",                   ":code:`fluid.BuildStrategy.ReduceStrategy`", ":code:`fluid.BuildStrategy.ReduceStrategy.AllReduce`", "使用数据并行训练模型时选用 :code:`AllReduce` 模式训练还是 :code:`Reduce` 模式训练."
    ":code:`enable_backward_optimizer_op_deps`", "bool", "FALSE", "在反向操作和参数更新操作之间添加依赖，保证在所有的反向操作都运行结束之后才开始运行参数更新操作."
    ":code:`fuse_all_optimizer_ops`",            "bool", "FALSE", "对模型中的参数更新算法进行融合."
    ":code:`fuse_all_reduce_ops`",               "bool", "FALSE", "多卡训练时，将all_reduce Op进行融合."
    ":code:`fuse_relu_depthwise_conv`",          "bool", "FALSE", "如果模型中存在relu和depthwise_conv，并且是连接的，即relu->depthwise_conv，该选项可以将这两个操作合并为一个."
    ":code:`fuse_broadcast_ops`",                "bool", "FALSE", "在 :code:`Reduce` 模式下，对最后的多个Broadcast操作融合为一个."
    ":code:`mkldnn_enabled_op_types`",           "list", "{}",    "如果是CPU训练，可以用 :code:`mkldnn_enabled_op_types` 指明模型中的那些操作可以使用MKLDNN库，如果不进行设置，模型可以使用MKLDNN库的所有操作都会使用MKLDNN库."

说明：
 - 关于 :code:`reduce_strategy` ，在 :code:`ParallelExecutor` 对于数据并行支持两种参数更新模式： :code:`AllReduce` 和 :code:`Reduce` 。在 :code:`AllReduce` 模式下，各个节点上计算得到梯度之后，调用 :code:`AllReduce` 操作，梯度在各个节点上聚合，然后各个节点分别进行参数更新。在 :code:`Reduce` 模式下，参数的更新操作被均匀的分配到各个节点上，即各个节点计算得到梯度之后，将梯度在指定的节点上进行 :code:`Reduce` ，然后在该节点上，最后将更新之后的参数Broadcast到其他节点。即：如果模型中有100个参数需要更新，训练时使用的是4个节点，在 :code:`AllReduce` 模式下，各个节点需要分别对这100个参数进行更新；在 :code:`Reduce` 模式下，各个节点需要分别对这25个参数进行更新，最后对更新的参数Broadcast到其他节点上.
 - 关于 :code:`enable_backward_optimizer_op_deps` ，在多卡训练时，打开该选项可能会提升训练速度.
 - 关于 :code:`fuse_all_optimizer_ops` ，目前只支持SGD、Adam和Momentum算法。**注意：目前不支持sparse参数梯度**。
 - 关于 :code:`fuse_all_reduce_ops` ，多GPU训练时，可以对 :code:`AllReduce` 操作进行融合，以减少 :code:`AllReduce` 的调用次数。默认情况下会将同一layer中参数的梯度的 :code:`AllReduce` 操作合并成一个，比如对于 :code:`fluid.layers.fc` 中有Weight和Bias两个参数，打开该选项之后，原本需要两次 :code:`AllReduce` 操作，现在只用一次 :code:`AllReduce` 操作。此外，为支持更大粒度的参数梯度融合，Paddle提供了 :code:`FLAGS_fuse_parameter_memory_size` 选项，用户可以指定融合AllReduce操作之后，每个 :code:`AllReduce` 操作的梯度字节数，比如希望每次 :code:`AllReduce` 调用传输64MB的梯度，:code:`export FLAGS_fuse_parameter_memory_size=64` 。**注意：目前不支持sparse参数梯度**。
 - 关于 :code:`mkldnn_enabled_op_types` ，支持mkldnn库的Op有：transpose, sum, softmax, requantize, quantize, pool2d, lrn, gaussian_random, fc, dequantize, conv2d_transpose, conv2d, conv3d, concat, batch_norm, relu, tanh, sqrt, abs. 

3.3 ExecutionStrategy中的配置参数
^^^^^^^^^^^^^^^^
ExecutionStrategy配置选项
>>>>>>>>>>>>>>>

..  csv-table:: 
    :header: "选项", "类型", "默认值", "说明"
    :widths: 3, 3, 5, 5

    ":code:`num_iteration_per_drop_scope`", "INT", "1", "经过多少次迭代之后清理一次local execution scope."
    ":code:`num_threads`",                  "INT", "对于CPU：2*dev_count；对于GPU：4*dev_count. （这是一个经验值）", ":code:`ParallelExecutor` 中执行所有Op使用的线程池大小"

说明：
 - 关于 :code:`num_iteration_per_drop_scope` ，框架在运行过程中会产生一些临时变量，这些变量被放在local execution scope中。通常每经过一个batch就要清理一下local execution scope中的变量，但是由于GPU是异步设备，在清理local execution scope之前需要对所有的GPU调用一次同步操作，因此耗费的时间较长。为此我们在 :code:`execution_strategy` 中添加了 :code:`num_iteration_per_drop_scope` 选项。用户可以指定经过多少次迭代之后清理一次local execution scope。
 - 关于 :code:`num_threads` ，":code:`ParallelExecutor` 中根据Op之间的依赖关系确定Op的执行顺序的，即Op的输入都已经变为ready状态之后，该Op会被放到一个队列中，等待被执行。 :code:`ParallelExecutor` 内部有一个任务调度线程和一个线程池，任务调度线程从队列中取出所有Ready的Op，并将其放到线程队列中。 :code:`num_threads` 表示线程池的大小。根据以往的经验，对于CPU任务，:code:`num_threads=2*dev_count` 时性能较好，对于GPU任务，:code:`num_threads=4*dev_count` 时性能较好。**注意：线程池不是越大越好**。

执行策略配置推荐
>>>>>>>>>>>>>>>

- 在显存足够的前提下，建议将 :code:`exec_strategy.num_iteration_per_drop_scope` 设置成一个较大的值，比如设置 :code:`exec_strategy.num_iteration_per_drop_scope=100` ，这样可以避免反复地申请和释放内存。该配置对于一些模型的优化效果较为明显。
- 对于一些较小的模型，比如mnist、language_model等，多个线程乱序调度op的开销大于其收益，因此推荐设置 :code:`exec_strategy.num_threads=1`  。

CPU训练设置
>>>>>>>>>>>>>>>

- 如果使用CPU做数据并行训练，需要指定环境变量CPU_NUM，这个环境变量指定程序运行过程中使用的 :code:`CPUPlace` 的个数。
- 如果使用CPU进行数据并行训练，并且 :code:`build_strategy.reduce_strategy` =  :code:`fluid.BuildStrategy.ReduceStrategy.Reduce` ，所有 :code:`CPUPlace` 上的参数是共享的，因此对于一些使用CPU进行数据并行训练的模型，选用 :code:`Reduce` 模式可能会更快一些。

4. 运行时FLAGS设置
=============
Fluid中有一些FLAGS可以有助于性能优化

- FLAGS_fraction_of_gpu_memory_to_use表示每次分配GPU显存的最小单位，取值范围为[0, 1)。由于CUDA原生的显存分配cuMalloc和释放cuFree操作均是同步操作，非常耗时，因此将FLAGS_fraction_of_gpu_memory_to_use设置成一个较大的值，比如0.92（默认值），可以显著地加速训练的速度。
- FLAGS_cudnn_exhaustive_search表示cuDNN在选取conv实现算法时采取穷举搜索策略，因此往往能选取到一个更快的conv实现算法，这对于CNN网络通常都是有加速的。但穷举搜索往往也会增加cuDNN的显存需求，因此用户可根据模型的实际情况选择是否设置该变量。
- FLAGS_enable_cublas_tensor_op_math表示是否使用TensorCore加速计算cuBLAS。这个环境变量只在Tesla V100以及更新的GPU上适用，且可能会带来一定的精度损失。

5. 使用Profile工具进行性能分析
=============

为方便用户更好的发现程序中的性能瓶颈，Paddle提供了多种Profile工具，这些工具的详细介绍和使用说明请参考 `性能调优 <http://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/advanced_usage/development/profiling/index_cn.html>`_ 。
