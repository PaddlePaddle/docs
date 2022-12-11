.. _api_guide_singlenode_training_best_practice:


#####################
单机训练优秀实践
#####################

开始优化您的单机训练任务
-------------------------

PaddlePaddle Fluid 可以支持在现代 CPU、GPU 平台上进行训练。如果您发现 Fluid 进行单机训练的速度较慢，您可以根据这篇文档的建议对您的 Fluid 程序进行优化。

神经网络训练代码通常由三个步骤组成：网络构建、数据准备、模型训练。这篇文档将分别从这三个方向介绍 Fluid 训练中常用的优化方法。


1. 网络构建过程中的配置优化
==================

这部分优化与具体的模型有关，在这里，我们列举出一些优化过程中遇到过的一些示例。

1.1 cuDNN 操作的选择
^^^^^^^^^^^^^^^^

cuDNN 是 NVIDIA 提供的深度神经网络计算库，其中包含了很多神经网络中常用算子，Paddle 中的部分 Op 底层调用的是 cuDNN 库，例如 :code:`conv2d` ：

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
                               name=None,
                               data_format="NCHW")

在 :code:`use_cudnn=True` 时，框架底层调用的是 cuDNN 中的卷积操作。

通常 cuDNN 库提供的操作具有很好的性能表现，其性能明显优于 Paddle 原生的 CUDA 实现，比如 :code:`conv2d` 。但是 cuDNN 中有些操作的性能较差，比如： :code:`conv2d_transpose` 在 :code:`batch_size=1` 时、:code:`pool2d` 在 :code:`global_pooling=True` 时等，这些情况下，cuDNN 实现的性能差于 Paddle 的 CUDA 实现，建议手动设置 :code:`use_cudnn=False` 。

1.2 减少模型中 Layer 的个数
^^^^^^^^^^^^^^^^^^

为方便用户使用，飞桨提供一些不同粒度的 Layer，其中有些 Layer 的组合可以通过单个 Layer 完成。比如：

(1) :code:`fluid.layers.softmax_with_cross_entropy` ，该操作其实是 :code:`fluid.layers.softmax` 和 :code:`fluid.layers.cross_entropy` 的组合，因此如果模型中有出现

.. code-block:: python

    logits = fluid.layers.softmax(logits)
    loss = fluid.layers.cross_entropy(logits, label, ignore_index=255)

可以直接替换成

.. code-block:: python

    loss = fluid.layers.softmax_with_cross_entropy(logits, label, ignore_index=255, numeric_stable_mode=True)


(2) 如果模型中需要对数据进行标准化，可以直接使用 :code:`fluid.layers.data_norm` ，而不用通过一系列 layer 组合出数据的标准化操作。

因此，建议在构建模型时优先使用飞桨提供的单个 Layer 完成所需操作，这样减少模型中 Layer 的个数，并因此加速模型训练。


2. 数据准备优化
=============

数据准备通常分为两部分：第一部分是数据加载，即程序从磁盘中加载训练/预测数据；第二部分是数据预处理，程序对加载的数据进行预处理，比如图像任务通常需要进行数据增强、Shuffle 等。
这两部分需要用户根据自己的模型需要进行设置，只需要最后得到 Data Reader 接口即可。Data Reader 返回 iterable 对象，可以每次返回一条样本或者一组样本。代码示例如下：

.. code-block:: python

    def data_reader(width, height):
        def reader():
            while True:
                yield np.random.uniform(-1, 1,size=width*height), np.random.randint(0,10)
        return reader
    train_data_reader = data_reader(32, 32)


Paddle 提供了两种方式从 Data Reader 中读取数据： :ref:`user_guide_use_numpy_array_as_train_data` 和 :ref:`user_guides_use_py_reader` ，详情请参考文档 :ref:`user_guide_prepare_data` 。

2.1 同步数据读取
^^^^^^^^^^^^^^^^

同步数据读取是一种简单并且直观的数据准备方式，代码示例如下：

.. code-block:: python

    image = fluid.data(name="image", shape=[None, 1, 28, 28], dtype="float32")
    label = fluid.data(name="label", shape=[None, 1], dtype="int64")
    # 模型定义
    # ……
    prediction = fluid.layers.fc(input=image, size=10)
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_loss = fluid.layers.mean(loss)
    # ……
    # 读取数据
    # paddle.dataset.mnist.train()返回数据读取的 Reader,每次可以从 Reader 中读取一条样本，batch_size 为 128
    train_reader = paddle.batch(paddle.dataset.mnist.train(), 128)

    # 读取数据
    end = time.time()
    for batch_id, batch in enumerate(train_reader):
        data_time = time.time() - end
        # 训练网络
        executor.run(feed={...}, fetch_list=[...])
        batch_time = time.time() - end
        end = time.time()


用户首先需要通过 :code:`fluid.data` 定义模型的输入，然后根据输入构建模型，最后从事先自定义的 Reader 函数中获取一个 batch 的数据，并将数据传递给执行器。

采用同步数据读取方式时，用户可通过加入 Python 计时函数 :code:`time.time()` 来统计数据准备部分和执行部分所占用的时间。
由于数据准备和执行是顺序进行的，所以程序的执行速度可能较慢。如果用户想进行模型调试的话，同步数据读取是一个不错的选择。


2.2 异步数据读取
^^^^^^^^^^^^^^^^

Paddle 里面使用 paddle.fluid.io. :ref:`cn_api_fluid_io_DataLoader` 接口来实现异步数据读取，代码示例如下：

.. code-block:: python

    image = fluid.data(name="image", shape=[None, 1, 28, 28], dtype="float32")
    label = fluid.data(name="label", shape=[None, 1], dtype="int64")
    dataloader = fluid.io.DataLoader.from_generator(
            feed_list=[image, label],
            capacity=64,
            iterable=False,
            use_double_buffer=True)
    # 模型定义
    # ……
    prediction = fluid.layers.fc(input=image, size=10)
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_loss = fluid.layers.mean(loss)
    # ……
    # 读取数据
    train_reader = paddle.batch(paddle.dataset.mnist.train(), 128)
    data_loader.set_batch_generator(train_reader, places=places)

    # 启动 data_loader
    data_loader.start()
    batch_id = 0
    try:
        end = time.time()
        while True:
            print("queue size: ", data_loader.queue.size())
            loss, = executor.run(fetch_list=[...])
            # ...
            batch_time = time.time() - end
            end = time.time()
            batch_id += 1
    except fluid.core.EOFException:
        data_loader.reset()

用户首先需要通过 :code:`fluid.io.DataLoader.from_generator` 定义 DataLoader 对象，并使用 :code:`set_batch_generator` 方法将自定义的 Reader 与 DataLoader 绑定。
若 DataLoader 被定义成不可迭代的（ :code:`iterable=False` ），在训练开始之前，通过调用 :code:`start()` 方法来启动数据读取。
在数据读取结束之后， :code:`executor.run` 会抛出 :code:`fluid.core.EOFException` ，表示训练已经遍历完 Reader 中的所有数据。

采用异步数据读取时，Python 端和 C++端共同维护一个数据队列，Python 端启动一个线程，负责向队列中插入数据，C++端在训练/预测过程中，从数据队列中获取数据，并将该数据从对队列中移除。
用户可以在程序运行过程中，监测数据队列是否为空，如果队列始终不为空，表明数据准备的速度比模型执行的速度快，这种情况下数据读取可能不是瓶颈。

另外，Paddle 提供的一些 FLAGS 也能很好的帮助分析性能。如果用户希望评估一下在完全没有数据读取开销情况下模型的性能，可以设置一下环境变量：:code:`FLAGS_reader_queue_speed_test_mode` ，在该变量为 True 情况下，C++端从数据队列中获取数据之后，不会从数据队列中移除，这样能够保证数据队列始终不为空，从而避免了 C++端读取数据时的等待开销。

**需要特别注意的是，** :code:`FLAGS_reader_queue_speed_test_mode` **只能在性能分析的时候打开，正常训练模型时需要关闭。**

为降低训练的整体时间，建议用户使用异步数据读取的方式，并开启 :code:`use_double_buffer=True` 。用户可根据模型的实际情况设置数据队列的大小。
如果数据准备的时间大于模型执行的时间，或者出现了数据队列为空的情况，就需要考虑对数据读取 Reader 进行加速。
常用的方法是 **使用 Python 多进程准备数据** ，一个简单的使用多进程准备数据的示例，可以参考 `YOLOv3 <https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/yolov3/reader.py>`_ 。

Python 端的数据预处理，都是使用 CPU 完成。如果 Paddle 提供了相应功能的 API，可将这部分预处理功能写到模型配置中，如此 Paddle 就可以使用 GPU 来完成该预处理功能，这样也可以减轻 CPU 预处理数据的负担，提升总体训练速度。

3. 模型训练相关优化
=============

3.1 执行器介绍
^^^^^^^^^^^^^^^^

目前 Paddle 的 Python API 中提供了 :code:`fluid.compiler.CompiledProgram` 的概念，用户可以通过 :code:`CompiledProgram` 将传入的 program 进行编译。
如果希望采用数据并行模式训练，只需要将 :code:`CompiledProgram` 返回的对象调用一下 :code:`with_data_parallel` 即可，最后统一通过 :code:`executor.run(…)` 执行 compiled_program。

虽然统一通过 :code:`executor.run(…)` 接口来执行，实际底层的执行策略有两种，对应 C++部分的两个执行器，即 :code:`Executor` 和 :code:`ParallelExecutor` ，如果用户采用数据并行模式，C++部分使用的是 :code:`ParallelExecutor` ，除此之外都是使用 :code:`Executor` 。
这两个执行器的差别：

..  csv-table::
    :header: "执行器 ", "执行对象", "执行策略"
    :widths: 3, 3, 5

    ":code:`Executor`",         ":code:`Program`",   "根据 :code:`Program` 中 Operator 定义的先后顺序依次运行。"
    ":code:`ParallelExecutor`", "SSA Graph", "根据 Graph 中各个节点之间的依赖关系，通过多线程运行。"


可以看出， :code:`Executor` 的内部逻辑非常简单，但性能可能会弱一些，因为 :code:`Executor` 对于 program 中的操作是串行执行的。
而 :code:`ParallelExecutor` 首先会将 program 转变为计算图，并分析计算图中节点间的连接关系，对图中没有相互依赖的节点（OP），通过多线程并行执行。

因此， :code:`Executor` 是一个轻量级的执行器，目前主要用于参数初始化、模型保存、模型加载。
:code:`ParallelExecutor` 是 :code:`Executor` 的升级版本，目前 :code:`ParallelExecutor` 主要用于模型训练，包括单机单卡、单机多卡以及多机多卡训练。

:code:`ParallelExecutor` 执行计算图之前，可以对计算图进行一些优化，比如使计算图中的一些操作是 In-place 的、将计算图中的参数更新操作进行融合等。
用户还可以调整 :code:`ParallelExecutor` 执行过程中的一些配置，比如执行计算图的线程数等。这些配置分别是构建策略（BuildStrategy）和执行策略（ExecutionStrategy）参数来设置的。


一个简单的使用示例如下：

.. code-block:: python

    build_strategy = fluid.BuildStrategy()
    build_strategy.enable_inplace = True
    build_strategy.fuse_all_optimizer_ops=True

    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = 4

    train_program = fluid.compiler.CompiledProgram(main_program).with_data_parallel(
                loss_name=loss.name,
                build_strategy=build_strategy,
                exec_strategy=exec_strategy)

    place = fluid.CUDAPlace(0)
    exe = Executor(place)
    # 使用 DataLoader 读取数据，因此执行时不需要设置 feed
    fetch_outs = exe.run(train_program, fetch_list=[loss.name])



3.2 构建策略（BuildStrategy）配置参数介绍
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

BuildStrategy 中提供了一些关于计算图优化的策略，这些策略可以在不同程度上提升模型的训练速度，但是其中一些策略与模型的结构有关，比如 :code:`fuse_all_optimizer_ops` 不支持 sparse 梯度，我们正在积极的完善这些策略，并在下一个版本将这些策略默认打开。

构建策略的详细介绍如下：

..  csv-table::
    :header: "选项", "类型", "默认值", "说明"
    :widths: 3, 3, 3, 5

    ":code:`reduce_strategy`",                   ":code:`fluid.BuildStrategy.ReduceStrategy`", ":code:`fluid.BuildStrategy.ReduceStrategy.AllReduce`", "使用数据并行训练模型时选用 :code:`AllReduce` 模式训练还是 :code:`Reduce` 模式训练。"
    ":code:`enable_backward_optimizer_op_deps`", "bool", "True", "在反向操作和参数更新操作之间添加依赖，保证在所有的反向操作都运行结束之后才开始运行参数更新操作。"
    ":code:`fuse_all_optimizer_ops`",            "bool", "False", "对模型中的参数更新算法进行融合。"
    ":code:`fuse_all_reduce_ops`",               "bool", "False", "多卡训练时，将 all_reduce 操作进行融合。"
    ":code:`fuse_relu_depthwise_conv`",          "bool", "False", "如果模型中存在 relu 和 depthwise_conv，并且是连接的，即 relu->depthwise_conv，该选项可以将这两个操作合并为一个。"
    ":code:`fuse_broadcast_ops`",                "bool", "False", "在 :code:`Reduce` 模式下，将最后的多个 Broadcast 操作融合为一个。"
    ":code:`mkldnn_enabled_op_types`",           "list", "{}",    "如果是 CPU 训练，可以用 :code:`mkldnn_enabled_op_types` 指明模型中的那些操作可以使用 MKLDNN 库。默认情况下，模型中用到的操作如果在 Paddle 目前支持的可以使用 mkldnn 库计算的列表中，这些操作都会调用 mkldnn 库的接口进行计算。"
    ":code:`debug_graphviz_path`",               "str",  "{}",    "将 Graph 以 graphviz 格式输出到 debug_graphviz_path 所指定的文件中。"

参数说明：

(1) 关于 :code:`reduce_strategy` ，在 :code:`ParallelExecutor` 对于数据并行支持两种参数更新模式： :code:`AllReduce` 和 :code:`Reduce` 。在 :code:`AllReduce` 模式下，各个节点上计算得到梯度之后，调用 :code:`AllReduce` 操作，梯度在各个节点上聚合，然后各个节点分别进行参数更新。在 :code:`Reduce` 模式下，参数的更新操作被均匀的分配到各个节点上，即各个节点计算得到梯度之后，将梯度在指定的节点上进行 :code:`Reduce` ，然后在该节点上，最后将更新之后的参数 Broadcast 到其他节点。即：如果模型中有 100 个参数需要更新，训练时使用的是 4 个节点，在 :code:`AllReduce` 模式下，各个节点需要分别对这 100 个参数进行更新；在 :code:`Reduce` 模式下，各个节点需要分别对这 25 个参数进行更新，最后将更新的参数 Broadcast 到其他节点上。注意：如果是使用 CPU 进行数据并行训练，在 Reduce 模式下，不同 CPUPlace 上的参数是共享的，所以在各个 CPUPlace 上完成参数更新之后不用将更新后的参数 Broadcast 到其他 CPUPlace。

(2) 关于 :code:`enable_backward_optimizer_op_deps` ，在多卡训练时，打开该选项可能会提升训练速度。

(3) 关于 :code:`fuse_all_optimizer_ops` ，目前只支持 SGD、Adam 和 Momentum 算法。 **注意：目前不支持 sparse 参数梯度** 。

(4) 关于 :code:`fuse_all_reduce_ops` ，多 GPU 训练时，可以对 :code:`AllReduce` 操作进行融合，以减少 :code:`AllReduce` 的调用次数。默认情况下会将同一 layer 中参数的梯度的 :code:`AllReduce` 操作合并成一个，比如对于 :code:`fluid.layers.fc` 中有 Weight 和 Bias 两个参数，打开该选项之后，原本需要两次 :code:`AllReduce` 操作，现在只用一次 :code:`AllReduce` 操作。此外，为支持更大粒度的参数梯度融合，Paddle 提供了 :code:`FLAGS_fuse_parameter_memory_size` 选项，用户可以指定融合 AllReduce 操作之后，每个 :code:`AllReduce` 操作的梯度字节数，比如希望每次 :code:`AllReduce` 调用传输 64MB 的梯度，:code:`export FLAGS_fuse_parameter_memory_size=64` 。 **注意：目前不支持 sparse 参数梯度** 。

(5) 关于 :code:`mkldnn_enabled_op_types` ，目前 Paddle 的 Op 中可以使用 mkldnn 库计算的操作包括：transpose、sum、softmax、requantize、quantize、pool2d、lrn、gaussian_random、fc、dequantize、conv2d_transpose、conv2d、conv3d、concat、batch_norm、relu、tanh、sqrt、abs。


3.3 执行策略（ExecutionStrategy）配置参数介绍
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ExecutionStrategy 中提供了关于计算图执行时的一些配置，这些配置可能会影响模型的训练速度。同时，这些配置与模型的结构有关，如果用户希望模型训练速度更快，可以调整一下这些配置。在后续的优化中，我们会对这部分进行优化，根据输入模型结构动态调整这些设置。

ExecutionStrategy 配置选项说明：

..  csv-table::
    :header: "选项", "类型", "默认值", "说明"
    :widths: 3, 3, 5, 5

    ":code:`num_iteration_per_drop_scope`", "INT", "100", "经过多少次迭代之后清理一次 local execution scope"
    ":code:`num_threads`",                  "INT", "对于 CPU：2*dev_count；对于 GPU：4*dev_count. （这是一个经验值）", ":code:`ParallelExecutor` 中执行所有 Op 使用的线程池大小"

说明：

(1) 关于 :code:`num_iteration_per_drop_scope` ，框架在运行过程中会产生一些临时变量，默认每经过一个 batch 就要清理一下临时变量。由于 GPU 是异步设备，在清理之前需要对所有的 GPU 调用一次同步操作，因此耗费的时间较长。为此我们在 execution_strategy 中添加了 :code:`num_iteration_per_drop_scope` 选项。用户可以指定经过多少次迭代之后清理一次。

(2) 关于 :code:`num_threads` ，:code:`ParallelExecutor` 根据 Op 之间的依赖关系确定 Op 的执行顺序，即：当 Op 的输入都已经变为 ready 状态之后，该 Op 会被放到一个队列中，等待被执行。 :code:`ParallelExecutor` 内部有一个任务调度线程和一个线程池，任务调度线程从队列中取出所有 Ready 的 Op，并将其放到线程队列中。 :code:`num_threads` 表示线程池的大小。根据以往的经验，对于 CPU 任务，:code:`num_threads=2*dev_count` 时性能较好，对于 GPU 任务，:code:`num_threads=4*dev_count` 时性能较好。 **注意：线程池不是越大越好** 。


4. 运行时 FLAGS 设置优化
=================

Paddle 中有一些 FLAGS 可以有助于性能优化：

(1) :code:`FLAGS_cudnn_exhaustive_search` 表示在调用 cuDNN 中的卷积操作时，根据输入数据的 shape 等信息，采取穷举搜索的策略从算法库中选取到更快的卷积算法，进而实现对模型中卷积操作的加速。需要注意的是：
    - 在搜索算法过程中需要使用较多的显存，如果用户的模型中卷积操作较多，或者 GPU 卡显存较小，可能会出现显存不足问题。
    - 通过穷举搜索选择好算法之后，该算法会进入 Cache，以便下次运行时，如果输入数据的 shape 等信息不变，直接使用 Cache 中算法。

(2) :code:`FLAGS_enable_cublas_tensor_op_math` 表示是否使用 TensorCore 加速 cuBLAS 等 NV 提供的库中的操作。需要注意的是，这个环境变量只在 Tesla V100 以及更新的 GPU 上适用，且可能会带来一定的精度损失，通常该损失不会影响模型的收敛性。


5. 优秀实践
=================

(1) 尽可能的使用飞桨提供的单个 layer 实现所需操作。
(2) 采用异步数据读取。
(3) 模型训练相关优化：

    - 使用 ParallelExecutor 作为底层执行器。单卡训练，也可以调用 with_data_parallel 方法。代码示例：

    .. code-block:: python

        compiled_prog = compiler.CompiledProgram(
                  fluid.default_main_program()).with_data_parallel(
                  loss_name=loss.name)

    - 如果模型中参数的梯度都是非 sparse 的，可以打开 fuse_all_optimizer_ops 选项，将多个参数更新操作融合为一个。
    - 如果是多卡训练，可以打开 enable_backward_optimizer_op_deps、fuse_all_reduce_ops 选项。如果想指定每次每次 AllReduce 操作的数据大小，可以设置 :code:`FLAGS_fuse_parameter_memory_size`，比如 :code:`export FLAGS_fuse_parameter_memory_size=1` ，表示每次 AllReduce 调用传输 1MB 的梯度。
    - 使用 CPU 做数据并行训练时，推荐使用 Reduce 模型，因为在使用 CPU 进行数据并行训练时，在 Reduce 模式下，不同 CPUPlace 上的参数是共享的，所以在各个 CPUPlace 上完成参数更新之后不用将更新后的参数 Broadcast 到其他 CPUPlace 上，这对提升速度也有很大帮助。
    - 如果是 Reduce 模式，可打开 fuse_broadcast_ops 选项。
    - 如果用户的模型较小，比如 mnist、language_model 等，可以将 num_threads 设为 1。
    - 在显存足够的前提下，建议将 :code:`exec_strategy.num_iteration_per_drop_scope` 设置成一个较大的值，比如设置为 100，这样可以避免反复地申请和释放内存。

目前我们正在推进这些配置自动化的工作：即根据输入的模型结构自动配置这些选项，争取在下一个版本中实现，敬请期待。

(4) FLAGS 设置

.. code-block:: bash

    FLAGS_cudnn_exhaustive_search = True
    FLAGS_enable_cublas_tensor_op_math = True


6. 使用 Profile 工具进行性能分析
======================

为方便用户更好的发现程序中的性能瓶颈，Paddle 提供了多种 Profile 工具，这些工具的详细介绍和使用说明请参考 :ref:`api_guide_analysis_tools` 。
