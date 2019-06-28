#################
 fluid
#################

.. _cn_api_fluid_BuildStrategy:

BuildStrategy
-------------------------------

.. py:class::  paddle.fluid.BuildStrategy

``BuildStrategy`` 使用户更精准地控制 ``ParallelExecutor`` 中SSA图的建造方法。可通过设置 ``ParallelExecutor`` 中的 ``BuildStrategy`` 成员来实现此功能。

**代码示例**

.. code-block:: python
    
    import paddle.fluid as fluid
    build_strategy = fluid.BuildStrategy()
    build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce


.. py:attribute:: debug_graphviz_path

str类型。它表明了以graphviz格式向文件中写入SSA图的路径，有利于调试。 默认值为""。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    build_strategy = fluid.BuildStrategy()
    build_strategy.debug_graphviz_path = ""


.. py:attribute:: enable_sequential_execution

类型是BOOL。 如果设置为True，则ops的执行顺序将与program中的执行顺序相同。 默认为False。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    build_strategy = fluid.BuildStrategy()
    build_strategy.enable_sequential_execution = True


.. py:attribute:: fuse_broadcast_ops
     
bool类型。它表明了是否融合（fuse）broadcast ops。值得注意的是，在Reduce模式中，融合broadcast ops可以使程序运行更快，因为这个过程等同于延迟执行所有的broadcast ops。在这种情况下，所有的nccl streams仅用于一段时间内的NCCLReduce操作。默认为False。
     
.. py:attribute:: fuse_elewise_add_act_ops

bool类型。它表明了是否融合（fuse）elementwise_add_op和activation_op。这会使整体执行过程更快一些。默认为False。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    build_strategy = fluid.BuildStrategy()
    build_strategy.fuse_elewise_add_act_ops = True


.. py:attribute:: fuse_relu_depthwise_conv

BOOL类型，fuse_relu_depthwise_conv指示是否融合relu和depthwise_conv2d，它会节省GPU内存并可能加速执行过程。 此选项仅适用于GPU设备。 默认为False。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    build_strategy = fluid.BuildStrategy()
    build_strategy.fuse_relu_depthwise_conv = True

.. py:attribute:: gradient_scale_strategy

str类型。在 ``ParallelExecutor`` 中，存在三种定义 *loss@grad* 的方式，分别为 ``CoeffNumDevice``, ``One`` 与 ``Customized``。默认情况下， ``ParallelExecutor`` 根据设备数目来设置 *loss@grad* 。如果你想自定义 *loss@grad* ，你可以选择 ``Customized`` 方法。默认为 ``CoeffNumDevice`` 。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    build_strategy = fluid.BuildStrategy()
    build_strategy.gradient_scale_strategy = True

.. py:attribute:: memory_optimize

bool类型。设为True时可用于减少总内存消耗。为实验性属性，一些变量可能会被优化策略重用/移除。如果你需要在使用该特征时获取某些变量，请把变量的persistable property设为True。默认为False。

.. py:attribute:: reduce_strategy

str类型。在 ``ParallelExecutor`` 中，存在两种减少策略（reduce strategy），即 ``AllReduce`` 和 ``Reduce`` 。如果你需要在所有执行场所上独立地进行参数优化，可以使用 ``AllReduce`` 。反之，如果使用 ``Reduce`` 策略，所有参数的优化将均匀地分配给不同的执行场所，随之将优化后的参数广播给其他执行场所。在一些模型中， ``Reduce`` 策略执行速度更快一些。默认值为 ``AllReduce`` 。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    build_strategy = fluid.BuildStrategy()
    build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce

.. py:attribute:: remove_unnecessary_lock

BOOL类型。如果设置为True, GPU操作中的一些锁将被释放，ParallelExecutor将运行得更快，默认为 True。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    build_strategy = fluid.BuildStrategy()
    build_strategy.remove_unnecessary_lock = True


.. py:attribute:: sync_batch_norm

类型为bool，sync_batch_norm表示是否使用同步的批正则化，即在训练阶段通过多个设备同步均值和方差。

当前的实现不支持FP16培训和CPU。仅在一台机器上进行同步式批正则，不适用于多台机器。

默认为 False。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    build_strategy = fluid.BuildStrategy()
    build_strategy.sync_batch_norm = True


.. _cn_api_fluid_CompiledProgram:

CompiledProgram
-------------------------------

.. py:class:: paddle.fluid.CompiledProgram(program_or_graph)

编译成一个用来执行的Graph。

1. 首先使用layers(网络层)创建程序。
2. （可选）可使用CompiledProgram来在运行之前优化程序。
3. 定义的程序或CompiledProgram由Executor运行。

CompiledProgram用于转换程序以进行各种优化。例如，

- 预先计算一些逻辑，以便每次运行更快。
- 转换Program，使其可以在多个设备中运行。
- 转换Program以进行优化预测或分布式训练。注意：此部分尚未完成。

**代码示例**

.. code-block:: python
        
        import paddle.fluid as fluid
        import paddle.fluid.compiler as compiler
        import numpy
        import os
     
        place = fluid.CUDAPlace(0) # fluid.CPUPlace()
        exe = fluid.Executor(place)
     
        data = fluid.layers.data(name='X', shape=[1], dtype='float32')
        hidden = fluid.layers.fc(input=data, size=10)
        loss = fluid.layers.mean(hidden)
        fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)
     
        fluid.default_startup_program().random_seed=1
        exe.run(fluid.default_startup_program())
        compiled_prog = compiler.CompiledProgram(
                 fluid.default_main_program())
     
        x = numpy.random.random(size=(10, 1)).astype('float32')
        loss_data, = exe.run(compiled_prog,
                             feed={"X": x},
                             fetch_list=[loss.name])
参数：
  - **program_or_graph** (Graph|Program): 如果它是Program，那么它将首先被降成一个graph，以便进一步优化。如果它是一个graph（以前可能优化过），它将直接用于进一步的优化。注意：只有使用 with_data_parallel 选项编译时才支持graph。

.. py:method:: with_data_parallel(loss_name=None, build_strategy=None, exec_strategy=None, share_vars_from=None, places=None)

配置Program使其以数据并行方式运行。

**代码示例**

.. code-block:: python
            
            import paddle.fluid as fluid
            import paddle.fluid.compiler as compiler
            import numpy
            import os
     
            use_cuda = True
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            #注意：如果你使用CPU运行程序，需要具体设置CPU_NUM，
            #否则fluid会把逻辑核的所有数目设为CPU_NUM，
            #在这种情况下，输入的batch size应大于CPU_NUM，
            #否则程序会异常中断。
            if not use_cuda:
                os.environ['CPU_NUM'] = str(2)
     
            exe = fluid.Executor(place)
     
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            hidden = fluid.layers.fc(input=data, size=10)
            loss = fluid.layers.mean(hidden)
            fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)
     
            fluid.default_startup_program().random_seed=1
            exe.run(fluid.default_startup_program())
            compiled_prog = compiler.CompiledProgram(
                     fluid.default_main_program()).with_data_parallel(
                              loss_name=loss.name)
     
            x = numpy.random.random(size=(10, 1)).astype('float32')
            loss_data, = exe.run(compiled_prog,
                                 feed={"X": x},
                                 fetch_list=[loss.name])
     
参数：
  - **loss_name** （str） - 损失函数名称必须在训练过程中设置。 默认None。
  - **build_strategy** （BuildStrategy） -  build_strategy用于构建图，因此它可以在具有优化拓扑的多个设备/核上运行。 有关更多信息，请参阅  ``fluid.BuildStrategy`` 。 默认None。
  - **exec_strategy** （ExecutionStrategy） -  exec_strategy用于选择执行图的方式，例如使用多少线程，每次清理临时变量之前进行的迭代次数。 有关更多信息，请参阅 ``fluid.ExecutionStrategy`` 。 默认None。
  - **share_vars_from** （CompiledProgram） - 如果有，此CompiledProgram将共享来自share_vars_from的变量。 share_vars_from指定的Program必须由此CompiledProgram之前的Executor运行，以便vars准备就绪。
  - **places** （list(CUDAPlace)|list(CPUPlace)|None） - 如果提供，则仅在给定位置编译程序。否则，编译时使用的位置由Executor确定，使用的位置由环境变量控制：如果使用GPU，则标记FLAGS_selected_gpus或CUDA_VISIBLE_DEVICES设备；如果使用CPU，则标记CPU_NUM。例如，如果要在GPU 0和GPU 1上运行，请设置places=[fluid.CUDAPlace(0), fluid.CUDAPlace(1)]。如果要在2个CPU核心上运行，请设置places=[fluid.CPUPlace()]*2。

返回: self

.. py:method:: with_inference_optimize(config)

添加预测优化。

参数：
  - **config** - 用于创建预测器的NativeConfig或AnalysisConfig的实例

返回: self


.. _cn_api_fluid_cpu_places:

cpu_places
-------------------------------

.. py:function:: paddle.fluid.cpu_places(device_count=None)

创建 ``fluid.CPUPlace`` 对象列表。

如果 ``device_count`` 为None，则设备数目将由环境变量 ``CPU_NUM`` 确定。如果未设置 ``CPU_NUM`` ，则设备数目将由 ``multiprocessing.cpu_count()`` 确定。

参数：
  - **device_count** (None|int) - 设备数目

返回: CPUPlace列表

返回类型：out (list(fluid.CPUPlace))

**代码示例**

.. code-block:: python

           cpu_places = fluid.cpu_places()


.. _cn_api_fluid_CPUPlace:

CPUPlace
-------------------------------

.. py:class:: paddle.fluid.CPUPlace


CPUPlace是设备的描述符。它代表一个CPU，可以访问CPUPlace对应的内存。

**代码示例**

.. code-block:: python

        cpu_place = fluid.CPUPlace()


.. _cn_api_fluid_create_lod_tensor:


create_lod_tensor
-------------------------------

.. py:function:: paddle.fluid.create_lod_tensor(data, recursive_seq_lens, place)


该函数从一个numpy数组，列表或者已经存在的lod tensor中创建一个lod tensor。

通过一下几步实现:

1. 检查length-based level of detail (LoD,长度为基准的细节层次)，或称recursive_sequence_lengths(递归序列长度)的正确性

2. 将recursive_sequence_lengths转化为offset-based LoD(偏移量为基准的LoD)

3. 把提供的numpy数组，列表或者已经存在的lod tensor复制到CPU或GPU中(依据执行场所确定)

4. 利用offset-based LoD来设置LoD

例如：
假如我们想用LoD Tensor来承载一词序列的数据，其中每个词由一个整数来表示。现在，我们意图创建一个LoD Tensor来代表两个句子，其中一个句子有两个词，另外一个句子有三个。那么数 ``data`` 可以是一个numpy数组，形状为（5,1）。同时， ``recursive_seq_lens`` 为 [[2, 3]]，表明各个句子的长度。这个长度为基准的 ``recursive_seq_lens`` 将在函数中会被转化为以偏移量为基准的 LoD [[0, 2, 5]]。

.. code-block:: python

        import paddle.fluid as fluid
        import numpy as np
     
        t = fluid.create_lod_tensor(np.ndarray([5, 30]), [[2, 3]], fluid.CPUPlace())

参考 :ref:`api_guide_tensor` 以获取更多关于LoD的信息。

参数:
  - **data** (numpy.ndarray|list|LoDTensor) – 容纳着待复制数据的一个numpy数组、列表或LoD Tensor
  - **recursive_seq_lens** (list) – 一组列表的列表， 表明了由用户指明的length-based level of detail信息
  - **place** (Place) – CPU或GPU。 指明返回的新LoD Tensor存储地点

返回: 一个fluid LoDTensor对象，包含数据和 ``recursive_seq_lens`` 信息











.. _cn_api_fluid_create_random_int_lodtensor:


create_random_int_lodtensor
-------------------------------

.. py:function:: paddle.fluid.create_random_int_lodtensor(recursive_seq_lens, base_shape, place, low, high)



该函数创建一个存储多个随机整数的LoD Tensor。

该函数是经常在书中出现的案例，所以我们根据新的API： ``create_lod_tensor`` 更改它然后放在LoD Tensor板块里来简化代码。

该函数实现以下功能：

1. 根据用户输入的length-based ``recursive_seq_lens`` （基于长度的递归序列长）和在 ``basic_shape`` 中的基本元素形状计算LoDTensor的整体形状
2. 由此形状，建立numpy数组
3. 使用API： ``create_lod_tensor`` 建立LoDTensor


假如我们想用LoD Tensor来承载一词序列，其中每个词由一个整数来表示。现在，我们意图创建一个LoD Tensor来代表两个句子，其中一个句子有两个词，另外一个句子有三个。那么 ``base_shape`` 为[1], 输入的length-based ``recursive_seq_lens`` 是 [[2, 3]]。那么LoDTensor的整体形状应为[5, 1]，并且为两个句子存储5个词。

参数:
    - **recursive_seq_lens** (list) – 一组列表的列表， 表明了由用户指明的length-based level of detail信息
    - **base_shape** (list) – LoDTensor所容纳的基本元素的形状
    - **place** (Place) –  CPU或GPU。 指明返回的新LoD Tensor存储地点
    - **low** (int) – 随机数下限
    - **high** (int) – 随机数上限

返回: 一个fluid LoDTensor对象，包含张量数据和 ``recursive_seq_lens`` 信息

**代码示例**

.. code-block:: python

        import paddle.fluid as fluid
     
        t = fluid.create_random_int_lodtensor(recursive_seq_lens=[[2, 3]],base_shape=[30], place=fluid.CPUPlace(), low=0, high=10)

.. _cn_api_fluid_cuda_pinned_places:

cuda_pinned_places
-------------------------------


.. py:function:: paddle.fluid.cuda_pinned_places(device_count=None)



创建 ``fluid.CUDAPinnedPlace`` 对象列表。

如果 ``device_count`` 为None，则设备数目将由环境变量 ``CPU_NUM`` 确定。如果未设置 ``CPU_NUM`` ，则设备数目将由 ``multiprocessing.cpu_count()`` 确定。

参数：
  - **device_count** (None|int) - 设备数目

返回: CUDAPinnedPlace对象列表

返回类型：out(list(fluid.CUDAPinnedPlace))

**代码示例**

.. code-block:: python

        cuda_pinned_places_cpu_num = fluid.cuda_pinned_places()
        # 或者
        cuda_pinned_places = fluid.cuda_pinned_places(1)

.. _cn_api_fluid_cuda_places:

cuda_places
-------------------------------

.. py:function:: paddle.fluid.cuda_places(device_ids=None)

创建 ``fluid.CUDAPlace`` 对象列表。



如果 ``device_ids`` 为None，则首先检查 ``FLAGS_selected_gpus`` 的环境变量。如果 ``FLAGS_selected_gpus=0,1,2`` ，则返回的列表将为[fluid.CUDAPlace(0), fluid.CUDAPlace(1), fluid.CUDAPlace(2)]。如果未设置标志 ``FLAGS_selected_gpus`` ，则将返回所有可见的GPU places。


如果 ``device_ids`` 不是None，它应该是GPU的设备ID。例如，如果 ``device_id=[0,1,2]`` ，返回的列表将是[fluid.CUDAPlace(0), fluid.CUDAPlace(1), fluid.CUDAPlace(2)]。

参数：
  - **device_ids** (None|list(int)|tuple(int)) - GPU的设备ID列表

返回: CUDAPlace列表

返回类型：out (list(fluid.CUDAPlace))

**代码示例**

.. code-block:: python

      cuda_places = fluid.cuda_places()

.. _cn_api_fluid_CUDAPinnedPlace:

CUDAPinnedPlace
-------------------------------

.. py:class:: paddle.fluid.CUDAPinnedPlace

CUDAPinnedPlace是一个设备描述符，它所指代的存储空间可以被GPU和CPU访问。

**代码示例**

.. code-block:: python

      place = fluid.CUDAPinnedPlace()

.. _cn_api_fluid_CUDAPlace:

CUDAPlace
-------------------------------

.. py:class:: paddle.fluid.CUDAPlace

CUDAPlace是一个设备描述符，它代表一个GPU，并且每个CUDAPlace有一个dev_id（设备id）来表明当前CUDAPlace代表的卡数。dev_id不同的CUDAPlace所对应的内存不可相互访问。

**代码示例**

.. code-block:: python

       gpu_place = fluid.CUDAPlace(0)




.. _cn_api_fluid_DataFeedDesc:

DataFeedDesc
-------------------------------

.. py:class:: paddle.fluid.DataFeedDesc(proto_file)

数据描述符，描述输入训练数据格式。

这个类目前只用于AsyncExecutor(有关类AsyncExecutor的简要介绍，请参阅注释)

DataFeedDesc应由来自磁盘的有效protobuf消息初始化。

可以参考 :code:`paddle/fluid/framework/data_feed.proto` 查看我们如何定义message

一段典型的message可能是这样的：

.. code-block:: python

    f = open("data.proto", "w")
    print >> f, 'name: "MultiSlotDataFeed"'
    print >> f, 'batch_size: 2'
    print >> f, 'multi_slot_desc {'
    print >> f, '    slots {'
    print >> f, '         name: "words"'
    print >> f, '         type: "uint64"'
    print >> f, '         is_dense: false'
    print >> f, '         is_used: true'
    print >> f, '     }'
    print >> f, '     slots {'
    print >> f, '         name: "label"'
    print >> f, '         type: "uint64"'
    print >> f, '         is_dense: false'
    print >> f, '         is_used: true'
    print >> f, '    }'
    print >> f, '}'
    f.close()
    data_feed = fluid.DataFeedDesc('data.proto')

但是，用户通常不应该关心消息格式;相反，我们鼓励他们在将原始日志文件转换为AsyncExecutor可以接受的训练文件的过程中，使用 :code:`Data Generator` 生成有效数据描述。

DataFeedDesc也可以在运行时更改。一旦你熟悉了每个字段的含义，您可以修改它以更好地满足您的需要。例如:

.. code-block:: python

    data_feed = fluid.DataFeedDesc('data.proto')
    data_feed.set_batch_size(128)
    data_feed.set_dense_slots('wd')  # 名为'wd'的slot将被设置为密集的
    data_feed.set_use_slots('wd')    # 名为'wd'的slot将被用于训练

    # 最后，可以打印变量详细信息便于排出错误

    print(data_feed.desc())


参数：
  - **proto_file** (string) - 包含数据feed中描述的磁盘文件


.. py:method:: set_batch_size(batch_size)

设置batch size，训练期间有效


参数：
  - batch_size：batch size

**代码示例：**

.. code-block:: python

    f = open("data.proto", "w")
    print >> f, 'name: "MultiSlotDataFeed"'
    print >> f, 'batch_size: 2'
    print >> f, 'multi_slot_desc {'
    print >> f, '    slots {'
    print >> f, '         name: "words"'
    print >> f, '         type: "uint64"'
    print >> f, '         is_dense: false'
    print >> f, '         is_used: true'
    print >> f, '     }'
    print >> f, '     slots {'
    print >> f, '         name: "label"'
    print >> f, '         type: "uint64"'
    print >> f, '         is_dense: false'
    print >> f, '         is_used: true'
    print >> f, '    }'
    print >> f, '}'
    f.close()
    data_feed = fluid.DataFeedDesc('data.proto')
    data_feed.set_batch_size(128)

.. py:method:: set_dense_slots(dense_slots_name)

指定slot经过设置后将变成密集的slot，仅在训练期间有效。

密集slot的特征将被输入一个Tensor，而稀疏slot的特征将被输入一个lodTensor


参数：
  - **dense_slots_name** : slot名称的列表，这些slot将被设置为密集的

**代码示例：**

.. code-block:: python

    f = open("data.proto", "w")
    print >> f, 'name: "MultiSlotDataFeed"'
    print >> f, 'batch_size: 2'
    print >> f, 'multi_slot_desc {'
    print >> f, '    slots {'
    print >> f, '         name: "words"'
    print >> f, '         type: "uint64"'
    print >> f, '         is_dense: false'
    print >> f, '         is_used: true'
    print >> f, '     }'
    print >> f, '     slots {'
    print >> f, '         name: "label"'
    print >> f, '         type: "uint64"'
    print >> f, '         is_dense: false'
    print >> f, '         is_used: true'
    print >> f, '    }'
    print >> f, '}'
    f.close()
    data_feed = fluid.DataFeedDesc('data.proto')
    data_feed.set_dense_slots(['words'])

.. note::

  默认情况下，所有slot都是稀疏的

.. py:method:: set_use_slots(use_slots_name)


设置一个特定的slot是否用于训练。一个数据集包含了很多特征，通过这个函数可以选择哪些特征将用于指定的模型。

参数：
  - **use_slots_name** :将在训练中使用的slot名列表

**代码示例：**

.. code-block:: python
    
    f = open("data.proto", "w")
    print >> f, 'name: "MultiSlotDataFeed"'
    print >> f, 'batch_size: 2'
    print >> f, 'multi_slot_desc {'
    print >> f, '    slots {'
    print >> f, '         name: "words"'
    print >> f, '         type: "uint64"'
    print >> f, '         is_dense: false'
    print >> f, '         is_used: true'
    print >> f, '     }'
    print >> f, '     slots {'
    print >> f, '         name: "label"'
    print >> f, '         type: "uint64"'
    print >> f, '         is_dense: false'
    print >> f, '         is_used: true'
    print >> f, '    }'
    print >> f, '}'
    f.close()
    data_feed = fluid.DataFeedDesc('data.proto')
    data_feed.set_use_slots(['words'])

.. note::

  默认值不用于所有slot


.. py:method:: desc()

返回此DataFeedDesc的protobuf信息

返回：一个message字符串

**代码示例：**

.. code-block:: python
    
    f = open("data.proto", "w")
    print >> f, 'name: "MultiSlotDataFeed"'
    print >> f, 'batch_size: 2'
    print >> f, 'multi_slot_desc {'
    print >> f, '    slots {'
    print >> f, '         name: "words"'
    print >> f, '         type: "uint64"'
    print >> f, '         is_dense: false'
    print >> f, '         is_used: true'
    print >> f, '     }'
    print >> f, '     slots {'
    print >> f, '         name: "label"'
    print >> f, '         type: "uint64"'
    print >> f, '         is_dense: false'
    print >> f, '         is_used: true'
    print >> f, '    }'
    print >> f, '}'
    f.close()
    data_feed = fluid.DataFeedDesc('data.proto')
    print(data_feed.desc())






.. _cn_api_fluid_DataFeeder:

DataFeeder
-------------------------------

.. py:class:: paddle.fluid.DataFeeder(feed_list, place, program=None)



``DataFeeder`` 负责将reader(读取器)返回的数据转成一种特殊的数据结构，使它们可以输入到 ``Executor`` 和 ``ParallelExecutor`` 中。
reader通常返回一个minibatch条目列表。在列表中每一条目都是一个样本（sample）,它是由具有一至多个特征的列表或元组组成的。


以下是简单用法：

.. code-block:: python

  import paddle.fluid as fluid
  place = fluid.CPUPlace()
  img = fluid.layers.data(name='image', shape=[1, 28, 28])
  label = fluid.layers.data(name='label', shape=[1], dtype='int64')
  feeder = fluid.DataFeeder([img, label], fluid.CPUPlace())
  result = feeder.feed([([0] * 784, [9]), ([1] * 784, [1])])

在多GPU模型训练时，如果需要提前分别向各GPU输入数据，可以使用 ``decorate_reader`` 函数。

.. code-block:: python

  import paddle
  import paddle.fluid as fluid

  place=fluid.CUDAPlace(0)
  data = fluid.layers.data(name='data', shape=[3, 224, 224], dtype='float32')
  label = fluid.layers.data(name='label', shape=[1], dtype='int64')

  feeder = fluid.DataFeeder(place=place, feed_list=[data, label])
  reader = feeder.decorate_reader(
        paddle.batch(paddle.dataset.flowers.train(), batch_size=16), multi_devices=False)



参数：
    - **feed_list** (list) – 向模型输入的变量表或者变量表名
    - **place** (Place) – place表明是向GPU还是CPU中输入数据。如果想向GPU中输入数据, 请使用 ``fluid.CUDAPlace(i)`` (i 代表 the GPU id)；如果向CPU中输入数据, 请使用  ``fluid.CPUPlace()``
    - **program** (Program) – 需要向其中输入数据的Program。如果为None, 会默认使用 ``default_main_program()``。 缺省值为None


抛出异常:
  - ``ValueError``  – 如果一些变量不在此 Program 中


**代码示例**

.. code-block:: python

  import numpy as np
  import paddle
  import paddle.fluid as fluid

  place = fluid.CPUPlace()

  def reader():
      yield [np.random.random([4]).astype('float32'), np.random.random([3]).astype('float32')],
  
  main_program = fluid.Program()
  startup_program = fluid.Program()
  
  with fluid.program_guard(main_program, startup_program):
        data_1 = fluid.layers.data(name='data_1', shape=[1, 2, 2])
        data_2 = fluid.layers.data(name='data_2', shape=[1, 1, 3])
        out = fluid.layers.fc(input=[data_1, data_2], size=2)
        # ...

  feeder = fluid.DataFeeder([data_1, data_2], place)
  
  exe = fluid.Executor(place)
  exe.run(startup_program)
  for data in reader():
      outs = exe.run(program=main_program,
                     feed=feeder.feed(data),
                     fetch_list=[out]))


.. py:method:: feed(iterable)


根据feed_list（数据输入表）和iterable（可遍历的数据）提供的信息，将输入数据转成一种特殊的数据结构，使它们可以输入到 ``Executor`` 和 ``ParallelExecutor`` 中。

参数:
  - **iterable** (list|tuple) – 要输入的数据

返回：  转换结果

返回类型: dict

**代码示例**

.. code-block:: python

    import numpy.random as random
    import paddle.fluid as fluid
     
    def reader(limit=5):
        for i in range(limit):
            yield random.random([784]).astype('float32'), random.random([1]).astype('int64'), random.random([256]).astype('float32')
     
    data_1 = fluid.layers.data(name='data_1', shape=[1, 28, 28])
    data_2 = fluid.layers.data(name='data_2', shape=[1], dtype='int64')
    data_3 = fluid.layers.data(name='data_3', shape=[16, 16], dtype='float32')
    feeder = fluid.DataFeeder(['data_1','data_2', 'data_3'], fluid.CPUPlace())
     
    result = feeder.feed(reader())


.. py:method:: feed_parallel(iterable, num_places=None)


该方法获取的多个minibatch，并把每个minibatch提前输入进各个设备中。

参数:
    - **iterable** (list|tuple) – 要输入的数据
    - **num_places** (int) – 设备数目。默认为None。

返回: 转换结果

返回类型: dict

.. note::
     设备（CPU或GPU）的数目必须等于minibatch的数目

**代码示例**

.. code-block:: python

    import numpy.random as random
    import paddle.fluid as fluid
     
    def reader(limit=10):
        for i in range(limit):
            yield [random.random([784]).astype('float32'), random.randint(10)],
     
    x = fluid.layers.data(name='x', shape=[1, 28, 28])
    y = fluid.layers.data(name='y', shape=[1], dtype='int64')
     
    feeder = fluid.DataFeeder(['x','y'], fluid.CPUPlace())
    place_num = 2
    places = [fluid.CPUPlace() for x in range(place_num)]
    data = []
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    program = fluid.CompiledProgram(fluid.default_main_program()).with_data_parallel(places=places)
    for item in reader():
        data.append(item)
        if place_num == len(data):
            exe.run(program=program, feed=list(feeder.feed_parallel(data, place_num)), fetch_list=[])
            data = []

.. py:method::  decorate_reader(reader, multi_devices, num_places=None, drop_last=True)



将reader返回的输入数据batch转换为多个mini-batch，之后每个mini-batch都会被输入进各个设备（CPU或GPU）中。

参数：
        - **reader** (fun) – 该参数是一个可以生成数据的函数
        - **multi_devices** (bool) – bool型，指明是否使用多个设备
        - **num_places** (int) – 如果 ``multi_devices`` 为 ``True`` , 可以使用此参数来设置GPU数目。如果 ``multi_devices`` 为 ``None`` ，该函数默认使用当前训练机所有GPU设备。默认为None。
        - **drop_last** (bool) – 如果最后一个batch的大小比 ``batch_size`` 要小，则可使用该参数来指明是否选择丢弃最后一个batch数据。 默认为 ``True``

返回：转换结果

返回类型: dict

抛出异常： ``ValueError`` – 如果 ``drop_last`` 值为False并且data batch与设备不匹配时，产生此异常

**代码示例**

.. code-block:: python

    import numpy.random as random
    import paddle
    import paddle.fluid as fluid
     
    def reader(limit=5):
        for i in range(limit):
            yield (random.random([784]).astype('float32'), random.random([1]).astype('int64')),
     
    place=fluid.CUDAPlace(0)
    data = fluid.layers.data(name='data', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
     
    feeder = fluid.DataFeeder(place=place, feed_list=[data, label])
    reader = feeder.decorate_reader(reader, multi_devices=False)
     
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    for data in reader():
        exe.run(feed=data)






.. _cn_api_fluid_default_main_program:

default_main_program
-------------------------------

.. py:function:: paddle.fluid.default_main_program()





此函数用于获取默认或全局main program(主程序)。该主程序用于训练和测试模型。

``fluid.layers`` 中的所有layer函数可以向 ``default_main_program`` 中添加operators（算子）和variables（变量）。

``default_main_program`` 是fluid的许多编程接口（API）的Program参数的缺省值。例如,当用户program没有传入的时候，
``Executor.run()`` 会默认执行 ``default_main_program`` 。


返回： main program

返回类型: Program

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
     
    # Sample Network:
    data = fluid.layers.data(name='image', shape=[3, 224, 224], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
     
    conv1 = fluid.layers.conv2d(data, 4, 5, 1, act=None)
    bn1 = fluid.layers.batch_norm(conv1, act='relu')
    pool1 = fluid.layers.pool2d(bn1, 2, 'max', 2)
    conv2 = fluid.layers.conv2d(pool1, 16, 5, 1, act=None)
    bn2 = fluid.layers.batch_norm(conv2, act='relu')
    pool2 = fluid.layers.pool2d(bn2, 2, 'max', 2)
     
    fc1 = fluid.layers.fc(pool2, size=50, act='relu')
    fc2 = fluid.layers.fc(fc1, size=102, act='softmax')
     
    loss = fluid.layers.cross_entropy(input=fc2, label=label)
    loss = fluid.layers.mean(loss)
    opt = fluid.optimizer.Momentum(
        learning_rate=0.1,
        momentum=0.9,
        regularization=fluid.regularizer.L2Decay(1e-4))
    opt.minimize(loss)
     
    print(fluid.default_main_program())







.. _cn_api_fluid_default_startup_program:




default_startup_program
-------------------------------

.. py:function:: paddle.fluid.default_startup_program()



该函数可以获取默认/全局 startup program (启动程序)。

``fluid.layers`` 中的layer函数会新建参数、readers(读取器)、NCCL句柄作为全局变量。

startup_program会使用内在的operators（算子）去初始化他们，并由layer函数将这些operators追加到startup program中。

该函数将返回默认的或当前的startup_program。用户可以使用 ``fluid.program_guard`` 去切换program。

返回: startup program

返回类型: Program

**代码示例：**

.. code-block:: python

        import paddle.fluid as fluid
     
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program=main_program, startup_program=startup_program):
            x = fluid.layers.data(name="x", shape=[-1, 784], dtype='float32')
            y = fluid.layers.data(name="y", shape=[-1, 1], dtype='int32')
            z = fluid.layers.fc(name="fc", input=x, size=10, act="relu")
     
            print("main program is: {}".format(fluid.default_main_program()))
            print("start up program is: {}".format(fluid.default_startup_program()))



.. _cn_api_fluid_DistributeTranspiler:

DistributeTranspiler
-------------------------------

.. py:class:: paddle.fluid.DistributeTranspiler (config=None)


该类可以把fluid program转变为分布式数据并行计算程序（distributed data-parallelism programs）,可以有Pserver和NCCL2两种模式。
当program在Pserver（全称：parameter server）模式下， ``main_program`` (主程序)转为使用一架远程parameter server(即pserver,参数服务器)来进行参数优化，并且优化图会被输入到一个pserver program中。
在NCCL2模式下，transpiler会在 ``startup_program`` 中附加一个 ``NCCL_ID`` 广播算子（broadcasting operators）来实现在该集群中所有工作结点共享 ``NCCL_ID`` 。
调用 ``transpile_nccl2`` 后， 你 **必须** 将 ``trainer_id`` , ``num_trainers`` 参数提供给 ``ParallelExecutor`` 来启动NCCL2分布式模式。




**代码示例**

.. code-block:: python

  x = fluid.layers.data(name='x', shape=[13], dtype='float32')
  y = fluid.layers.data(name='y', shape=[1], dtype='float32')
  y_predict = fluid.layers.fc(input=x, size=1, act=None)
  
  cost = fluid.layers.square_error_cost(input=y_predict, label=y)
  avg_loss = fluid.layers.mean(cost)
  
  sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
  sgd_optimizer.minimize(avg_loss)

  #pserver模式下
  pserver_endpoints = "192.168.0.1:6174,192.168.0.2:6174"
  trainer_endpoints = "192.168.0.1:6174,192.168.0.2:6174"
  current_endpoint = "192.168.0.1:6174"
  trainer_id = 0
  trainers = 4
  role = "PSERVER"

  t = fluid.DistributeTranspiler()
  t.transpile(trainer_id, pservers=pserver_endpoints, trainers=trainers)
  if role == "PSERVER":
     pserver_program = t.get_pserver_program(current_endpoint)
     pserver_startup_program = t.get_startup_program(current_endpoint, pserver_program)
  elif role == "TRAINER":
     trainer_program = t.get_trainer_program()

  # nccl2模式下
  trainer_num = 2
  trainer_id = 0
  config = fluid.DistributeTranspilerConfig()
  config.mode = "nccl2"
  trainer_endpoints = "192.168.0.1:6174,192.168.0.2:6174"
  t = fluid.DistributeTranspiler(config=config)
  t.transpile(trainer_id=trainer_id, trainers=trainer_endpoints, current_endpoint="192.168.0.1:6174")
  exe = fluid.ParallelExecutor(
     loss_name=avg_loss.name,
     num_trainers=len(trainer_num,
     trainer_id=trainer_id
  )



.. py:method:: transpile(trainer_id, program=None, pservers='127.0.0.1:6174', trainers=1, sync_mode=True, startup_program=None, current_endpoint='127.0.0.1:6174')

该方法可以运行该transpiler（转译器）。转译输入程序。

参数:
  - **trainer_id** (int) – 当前Trainer worker的id, 如果有n个Trainer worker, id 取值范围为0 ~ n-1
  - **program** (Program|None) – 待transpile（转译）的program, 缺省为 ``fluid.default_main_program()``
  - **startup_program** (Program|None) - 要转译的 ``startup_program`` ,默认为 ``fluid.default_startup_program()``
  - **pservers** (str) – 内容为Pserver列表的字符串，格式为：按逗号区分不同的Pserver，每个Pserver的格式为 *ip地址:端口号*
  - **trainers** (int|str) – 在Pserver模式下，该参数指Trainer机的个数；在nccl2模式下，它是一个内容为Trainer终端列表的字符串
  - **sync_mode** (bool) – 是否做同步训练(synchronous training), 默认为True
  - **startup_program** (Program|None) – 待transpile（转译）的startup_program，默认为 ``fluid.default_main_program()``
  - **current_endpoint** (str) – 当需要把program转译（transpile）至NCCL2模式下时，需要将当前endpoint（终端）传入该参数。Pserver模式不使用该参数

**代码示例**

.. code-block:: python

    transpiler = fluid.DistributeTranspiler()
    t.transpile(
        trainer_id=0,
        pservers="127.0.0.1:7000,127.0.0.1:7001",
        trainers=2,
        sync_mode=False,
        current_endpoint="127.0.0.1:7000")



.. py:method:: get_trainer_program(wait_port=True)


该方法可以得到Trainer侧的program。

返回: Trainer侧的program

返回类型: Program

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    #this is an example, find available endpoints in your case
    pserver_endpoints = "192.168.0.1:6174,192.168.0.2:6174"
    trainer_id = 0
    trainers = 4
    t = fluid.DistributeTranspiler()
    t.transpile(trainer_id, trainers=trainers, pservers=pserver_endpoints)
    trainer_program = t.get_trainer_program()


.. py:method:: get_pserver_program(endpoint)


该方法可以得到Pserver（参数服务器）侧的程序

参数:
  - **endpoint** (str) – 当前Pserver终端

返回: 当前Pserver需要执行的program

返回类型: Program

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    #this is an example, find available endpoints in your case
    pserver_endpoints = "192.168.0.1:6174,192.168.0.2:6174"
    current_endpoint = "192.168.0.1:6174"
    trainer_id = 0
    trainers = 4
    t = fluid.DistributeTranspiler()
    t.transpile(
         trainer_id, pservers=pserver_endpoints, trainers=trainers)
    pserver_program = t.get_pserver_program(current_endpoint)


.. py:method:: get_pserver_programs(endpoint)


该方法可以得到Pserver侧用于分布式训练的 ``main_program`` 和 ``startup_program`` 。

参数:
  - **endpoint** (str) – 当前Pserver终端

返回: (main_program, startup_program), “Program”类型的元组

返回类型: tuple

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    #this is an example, find available endpoints in your case
    pserver_endpoints = "192.168.0.1:6174,192.168.0.2:6174"
    current_endpoint = "192.168.0.1:6174"
    trainer_id = 0
    trainers = 4
    t = fluid.DistributeTranspiler()
    t.transpile(
         trainer_id, pservers=pserver_endpoints, trainers=trainers)
    pserver_program, pserver_startup_program = t.get_pserver_programs(current_endpoint)



.. py:method:: get_startup_program(endpoint, pserver_program=None, startup_program=None)


**该函数已停止使用**
获取当前Pserver的startup_program，如果有多个被分散到不同blocks的变量，则修改operator的输入变量。

参数:
  - **endpoint** (str) – 当前Pserver终端
  - **pserver_program** (Program) – 已停止使用。 先调用get_pserver_program
  - **startup_program** (Program) – 已停止使用。应在初始化时传入startup_program

返回: Pserver侧的startup_program

返回类型: Program

**代码示例**

.. code-block:: python

    pserver_endpoints = "192.168.0.1:6174,192.168.0.2:6174"
    trainer_endpoints = "192.168.0.1:6174,192.168.0.2:6174"
    current_endpoint = "192.168.0.1:6174"
    trainer_id = 0
    trainers = 4
     
    t = fluid.DistributeTranspiler()
    t.transpile(trainer_id, pservers=pserver_endpoints, trainers=trainers)
    pserver_program = t.get_pserver_program(current_endpoint)
    pserver_startup_program = t.get_startup_program(current_endpoint,
                                                    pserver_program)
     





.. _cn_api_fluid_DistributeTranspilerConfig:

DistributeTranspilerConfig
-------------------------------

.. py:class:: paddle.fluid.DistributeTranspilerConfig


.. py:attribute:: slice_var_up (bool)

为多个Pserver（parameter server）将tensor切片, 默认为True。

.. py:attribute:: split_method (PSDispatcher)

可使用 RoundRobin 或者 HashName。

注意: 尝试选择最佳方法来达到Pserver间负载均衡。

.. py:attribute:: min_block_size (int)

block中分割(split)出的元素个数的最小值。

注意: 根据：`issuecomment-369912156 <https://github.com/PaddlePaddle/Paddle/issues/8638#issuecomment-369912156>`_ , 当数据块大小超过2MB时，我们可以有效地使用带宽。如果你想更改它，请详细查看 ``slice_variable`` 函数。

**代码示例**

.. code-block:: python
    
    config = fluid.DistributeTranspilerConfig()
    config.slice_var_up = True




.. _cn_api_fluid_ExecutionStrategy:

ExecutionStrategy
-------------------------------

.. py:class:: paddle.fluid.ExecutionStrategy

``ExecutionStrategy`` 允许用户更加精准地控制program在 ``ParallelExecutor`` 中的运行方式。可以通过在 ``ParallelExecutor`` 中设置本成员来实现。

**代码示例**

.. code-block:: python

    x = fluid.layers.data(name='x', shape=[13], dtype='float32')
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')
    y_predict = fluid.layers.fc(input=x, size=1, act=None)
     
    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_loss = fluid.layers.mean(cost)
     
    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    sgd_optimizer.minimize(avg_loss)

    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = 4

    train_exe = fluid.ParallelExecutor(use_cuda=False,
                                       loss_name=avg_loss.name,
                                     exec_strategy=exec_strategy)



.. py:attribute:: allow_op_delay

这是一个bool类型成员，表示是否推迟communication operators(交流运算)的执行，这样做会使整体执行过程更快一些。但是在一些模型中，allow_op_delay会导致程序中断。默认为False。



.. py:attribute:: num_iteration_per_drop_scope

int型成员。它表明了清空执行时产生的临时变量需要的程序执行迭代次数。因为临时变量的形状可能在两次重复过程中保持一致，所以它会使整体执行过程更快。默认值为1。

.. note::
  1. 如果在调用 ``run`` 方法时获取结果数据，``ParallelExecutor`` 会在当前程序重复执行尾部清空临时变量

  2. 在一些NLP模型里，该成员会致使GPU内存不足。此时，你应减少 ``num_iteration_per_drop_scope`` 的值

.. py:attribute:: num_iteration_per_run
它配置了当用户在python脚本中调用pe.run()时执行器会执行的迭代次数。

.. py:attribute:: num_threads

int型成员。它代表了线程池(thread pool)的大小。这些线程会被用来执行当前 ``ParallelExecutor`` 的program中的operator（算子，运算）。如果 :math:`num\_threads=1` ，则所有的operator将一个接一个地执行，但在不同的程序重复周期(iterations)中执行顺序可能不同。如果该成员没有被设置，则在 ``ParallelExecutor`` 中，它会依据设备类型(device type)、设备数目(device count)而设置为相应值。对GPU，:math:`num\_threads=device\_count∗4` ；对CPU， :math:`num\_threads=CPU\_NUM∗4` 。在 ``ParallelExecutor`` 中有关于 :math:`CPU\_NUM` 的详细解释。如果没有设置 :math:`CPU\_NUM` ， ``ParallelExecutor`` 可以通过调用 ``multiprocessing.cpu_count()`` 获取CPU数目(cpu count)。默认值为0。












.. _cn_api_fluid_executor:

Executor
-------------------------------


.. py:class:: paddle.fluid.Executor (place)




执行引擎（Executor）使用python脚本驱动，支持在单/多GPU、单/多CPU环境下运行。
Python Executor可以接收传入的program,并根据feed map(输入映射表)和fetch_list(结果获取表)
向program中添加feed operators(数据输入算子)和fetch operators（结果获取算子)。
feed map为该program提供输入数据。fetch_list提供program训练结束后用户预期的变量（或识别类场景中的命名）。

应注意，执行器会执行program中的所有算子而不仅仅是依赖于fetch_list的那部分。

Executor将全局变量存储到全局作用域中，并为临时变量创建局部作用域。
当每一mini-batch上的前向/反向运算完成后，局部作用域的内容将被废弃，
但全局作用域中的变量将在Executor的不同执行过程中一直存在。


**示例代码**

.. code-block:: python
    
    import paddle.fluid as fluid
    import paddle.fluid.compiler as compiler
    import numpy
    import os

    use_cuda = True
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(train_program, startup_program):
        data = fluid.layers.data(name='X', shape=[1], dtype='float32')
        hidden = fluid.layers.fc(input=data, size=10)
        loss = fluid.layers.mean(hidden)
        fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)

    # 仅运行一次startup program
    # 不需要优化/编译这个startup program
    startup_program.random_seed=1
    exe.run(startup_program)

    # 无需编译，直接运行main program
    x = numpy.random.random(size=(10, 1)).astype('float32')
    loss_data, = exe.run(train_program,
                     feed={"X": x},
                     fetch_list=[loss.name])

    # 另一种方法是，编译这个main program然后运行。
    # 参考CompiledProgram以获取更多信息。
    # 注意：如果你使用CPU运行程序，需要具体设置CPU_NUM，
    # 否则fluid会把逻辑核的所有数目设为CPU_NUM，
    # 在这种情况下，输入的batch size应大于CPU_NUM，
    # 否则程序会异常中断。
    if not use_cuda:
        os.environ['CPU_NUM'] = str(2)

    compiled_prog = compiler.CompiledProgram(
        train_program).with_data_parallel(
        loss_name=loss.name)
    loss_data, = exe.run(compiled_prog,
                         feed={"X": x},
                         fetch_list=[loss.name])


参数:
    - **place** (fluid.CPUPlace|fluid.CUDAPlace(n)) – 指明了 ``Executor`` 的执行场所



.. py:method:: close()


关闭这个执行器(Executor)。

调用这个方法后不可以再使用这个执行器。 对于分布式训练, 该函数会释放在PServers上和目前Trainer有关联的资源。

**示例代码**

.. code-block:: python
    
    import paddle.fluid as fluid

    cpu = fluid.CPUPlace()
    exe = fluid.Executor(cpu)
    # 执行训练或测试过程
    exe.close()


.. py:method:: run(program=None, feed=None, fetch_list=None, feed_var_name='feed', fetch_var_name='fetch', scope=None, return_numpy=True,use_program_cache=False)


调用该执行器对象的此方法可以执行program。通过feed map提供待学习数据，以及借助fetch_list得到相应的结果。
Python执行器(Executor)可以接收传入的program,并根据输入映射表(feed map)和结果获取表(fetch_list)
向program中添加数据输入算子(feed operators)和结果获取算子（fetch operators)。
feed map为该program提供输入数据。fetch_list提供program训练结束后用户预期的变量（或识别类场景中的命名）。

应注意，执行器会执行program中的所有算子而不仅仅是依赖于fetch_list的那部分。

**示例代码**

.. code-block:: python

            import paddle.fluid as fluid
            import numpy
     
            #首先创建执行引擎
            place = fluid.CPUPlace() # fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
     
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            hidden = fluid.layers.fc(input=data, size=10)
            loss = fluid.layers.mean(hidden)
            adam = fluid.optimizer.Adam()
            adam.minimize(loss)
     
            #仅运行startup程序一次
            exe.run(fluid.default_startup_program())

            x = numpy.random.random(size=(10, 1)).astype('float32')
            outs = exe.run(feed={'X': x},
                           fetch_list=[loss.name])
                           
参数：  
  - **program** (Program|CompiledProgram) – 需要执行的program,如果没有给定那么默认使用default_main_program (未编译的)
  - **feed** (dict) – 前向输入的变量，数据,词典dict类型, 例如 {“image”: ImageData, “label”: LabelData}
  - **fetch_list** (list) – 用户想得到的变量或者命名的列表, 该方法会根据这个列表给出结果
  - **feed_var_name** (str) – 前向算子(feed operator)变量的名称
  - **fetch_var_name** (str) – 结果获取算子(fetch operator)的输出变量名称
  - **scope** (Scope) – 执行这个program的域，用户可以指定不同的域。缺省为全局域
  - **return_numpy** (bool) – 如果为True,则将结果张量（fetched tensor）转化为numpy
  - **use_program_cache** (bool) – 是否跨批使用缓存程序设置。设置为True时，只有当（1）程序没有用数据并行编译，并且（2）program、 feed变量名和fetch_list变量名与上一步相比没有更改时，运行速度才会更快。
  
返回: 根据fetch_list来获取结果

返回类型: list(numpy.array)


.. py:method:: infer_from_dataset(program=None, dataset=None, scope=None, thread=0, debug=False, fetch_list=None, fetch_info=None, print_period=100)

infer_from_dataset的文档与train_from_dataset几乎完全相同，只是在分布式训练中，推进梯度将在infer_from_dataset中禁用。 infer_from_dataset（）可以非常容易地用于多线程中的评估。

参数：  
  - **program** (Program|CompiledProgram) – 需要执行的program,如果没有给定那么默认使用default_main_program (未编译的)
  - **dataset** (paddle.fluid.Dataset) – 在此函数外创建的数据集，用户应当在调用函数前提供完整定义的数据集。必要时请检查Dataset文件。默认为None
  - **scope** (Scope) – 执行这个program的域，用户可以指定不同的域。默认为全局域
  - **thread** (int) – 用户想要在这个函数中运行的线程数量。线程的实际数量为min(Dataset.thread_num, thread)，如果thread > 0，默认为0
  - **debug** (bool) – 是否开启debug模式，默认为False
  - **fetch_list** (Variable List) – 返回变量列表，每个变量都会在训练过程中被打印出来，默认为None
  - **fetch_info** (String List) – 每个变量的打印信息，默认为None
  - **print_period** (int) – 每两次打印之间间隔的mini-batches的数量，默认为100

返回: None

**示例代码**

.. code-block:: python

  import paddle.fluid as fluid
  place = fluid.CPUPlace() # 使用GPU时可设置place = fluid.CUDAPlace(0)
  exe = fluid.Executor(place)
  x = fluid.layers.data(name="x", shape=[10, 10], dtype="int64")
  y = fluid.layers.data(name="y", shape=[1], dtype="int64", lod_level=1)
  dataset = fluid.DatasetFactory().create_dataset()
  dataset.set_use_var([x, y])
  dataset.set_thread(1)
  filelist = [] # 您可以设置您自己的filelist，如filelist = ["dataA.txt"]
  dataset.set_filelist(filelist)
  exe.run(fluid.default_startup_program())
  exe.infer_from_dataset(program=fluid.default_main_program(),dataset=dataset)
     

.. py:method:: train_from_dataset(program=None, dataset=None, scope=None, thread=0, debug=False, fetch_list=None, fetch_info=None, print_period=100)

从预定义的数据集中训练。 数据集在paddle.fluid.dataset中定义。 给定程序（或编译程序），train_from_dataset将使用数据集中的所有数据样本。 输入范围可由用户给出。 默认情况下，范围是global_scope()。训练中的线程总数是thread。 训练中使用的线程数将是数据集中threadnum的最小值，同时也是此接口中线程的值。 可以设置debug，以便执行器显示所有算子的运行时间和当前训练任务的吞吐量。

注意：train_from_dataset将销毁每次运行在executor中创建的所有资源。

参数：  
  - **program** (Program|CompiledProgram) – 需要执行的program,如果没有给定那么默认使用default_main_program (未编译的)
  - **dataset** (paddle.fluid.Dataset) – 在此函数外创建的数据集，用户应当在调用函数前提供完整定义的数据集。必要时请检查Dataset文件。默认为None
  - **scope** (Scope) – 执行这个program的域，用户可以指定不同的域。默认为全局域
  - **thread** (int) – 用户想要在这个函数中运行的线程数量。线程的实际数量为min(Dataset.thread_num, thread)，如果thread > 0，默认为0
  - **debug** (bool) – 是否开启debug模式，默认为False
  - **fetch_list** (Variable List) – 返回变量列表，每个变量都会在训练过程中被打印出来，默认为None
  - **fetch_info** (String List) – 每个变量的打印信息，默认为None
  - **print_period** (int) – 每两次打印之间间隔的mini-batches的数量，默认为100

返回: None

**示例代码**

.. code-block:: python

        import paddle.fluid as fluid

        place = fluid.CPUPlace() # 通过设置place = fluid.CUDAPlace(0)使用GPU
        exe = fluid.Executor(place)
        x = fluid.layers.data(name="x", shape=[10, 10], dtype="int64")
        y = fluid.layers.data(name="y", shape=[1], dtype="int64", lod_level=1)
        dataset = fluid.DatasetFactory().create_dataset()
        dataset.set_use_var([x, y])
        dataset.set_thread(1)
        filelist = [] # 您可以设置您自己的filelist，如filelist = ["dataA.txt"]
        dataset.set_filelist(filelist)
        exe.run(fluid.default_startup_program())
        exe.infer_from_dataset(program=fluid.default_main_program(),
                               dataset=dataset)


.. _cn_api_fluid_global_scope:

global_scope
-------------------------------

.. py:function:: paddle.fluid.global_scope()


获取全局/默认作用域实例。很多api使用默认 ``global_scope`` ，例如 ``Executor.run`` 。

**示例代码**

.. code-block:: python

        import paddle.fluid as fluid
        import numpy

        fluid.global_scope().var("data").get_tensor().set(numpy.ones((2, 2)), fluid.CPUPlace())
        numpy.array(fluid.global_scope().find_var("data").get_tensor())

返回：全局/默认作用域实例

返回类型：Scope






.. _cn_api_fluid_gradients:

gradients
-------------------------------

.. py:function:: paddle.fluid.gradients(targets, inputs, target_gradients=None, no_grad_set=None)

将目标梯度反向传播到输入。

参数：  
  - **targets** (Variable|list[Variable]) – 目标变量
  - **inputs** (Variable|list[Variable]) – 输入变量
  - **target_gradients** (Variable|list[Variable]|None) – 目标的梯度变量，应与目标变量形状相同；如果设置为None，则以1初始化所有梯度变量
  - **no_grad_sethread** (set[string]) – 在Block 0中不具有梯度的变量，所有block中被设置 ``stop_gradient=True`` 的变量将被自动加入该set


返回：数组，包含与输入对应的梯度。如果一个输入不影响目标函数，则对应的梯度变量为None

返回类型：(list[Variable])

**示例代码**

.. code-block:: python

            import paddle.fluid as fluid

            x = fluid.layers.data(name='x', shape=[2,8,8], dtype='float32')
            x.stop_gradient=False
            y = fluid.layers.conv2d(x, 4, 1, bias_attr=False)
            y = fluid.layers.relu(y)
            y = fluid.layers.conv2d(y, 4, 1, bias_attr=False)
            y = fluid.layers.relu(y)
            z = fluid.gradients([y], x)
            print(z)



.. _cn_api_fluid_in_dygraph_mode:

in_dygraph_mode
-------------------------------

.. py:function:: paddle.fluid.in_dygraph_mode()

检查程序状态(tracer) - 是否在dygraph模式中运行

返回：如果Program是在动态图模式下运行的则为True。

返回类型：out(boolean)

**示例代码**

.. code-block:: python

    if fluid.in_dygraph_mode():
        pass


.. _cn_api_fluid_LoDTensor:

LoDTensor
-------------------------------

.. py:class:: paddle.fluid.LoDTensor


LoDTensor是一个具有LoD信息的张量(Tensor)

``np.array(lod_tensor)`` 可以将LoDTensor转换为numpy array。

``lod_tensor.lod()`` 可以获得LoD信息。

LoD是多层序列（Level of Details）的缩写，通常用于不同长度的序列。如果您不需要了解LoD信息，可以跳过下面的注解。

举例:

X 为 LoDTensor，它包含两个序列。第一个长度是2，第二个长度是3。

从Lod中可以计算出X的第一维度为5， 因为5=2+3， 说明X中有5个序列。在X中的每个序列中的每个元素有2列，因此X的shape为[5,2]。

::

  x.lod  =  [[2, 3]] 
  
  x.data = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]

  x.shape = [5, 2]


LoD可以有多个level(例如，一个段落可以有多个句子，一个句子可以有多个单词)。下面的例子中，Y为LoDTensor ，lod_level为2。表示有2个序列，第一个序列的长度是2(有2个子序列)，第二个序列的长度是1。第一序列的两个子序列长度分别为2和2。第二个序列的子序列的长度是3。


::
  
  y.lod = [[2 1], [2 2 3]]

  y.shape = [2+2+3, ...]

**示例代码**

.. code-block:: python

      import paddle.fluid as fluid
     
      t = fluid.LoDTensor()

.. note::

  在上面的描述中，LoD是基于长度的。在paddle内部实现中，lod是基于偏移的。因此,在内部,y.lod表示为[[0,2,3]，[0,2,4,7]](基于长度的Lod表示为为[[2-0,3-2]，[2-0,4-2,7-4]])。

  可以将LoD理解为recursive_sequence_length（递归序列长度）。此时，LoD必须是基于长度的。由于历史原因。当LoD在API中被称为lod时，它可能是基于偏移的。用户应该注意。




.. py:method:: has_valid_recursive_sequence_lengths(self: paddle.fluid.core.LoDTensor) → bool

检查LoDTensor的lod值的正确性。

返回:    是否带有正确的lod值

返回类型:    out (bool)

**示例代码**

.. code-block:: python
            
            import paddle.fluid as fluid
            import numpy as np
     
            t = fluid.LoDTensor()
            t.set(np.ndarray([5, 30]), fluid.CPUPlace())
            t.set_recursive_sequence_lengths([[2, 3]])
            print(t.has_valid_recursive_sequence_lengths()) # True

.. py:method::  lod(self: paddle.fluid.core_avx.LoDTensor) → List[List[int]]

得到LoD Tensor的LoD。

返回：LoD Tensor的LoD。

返回类型：out（List [List [int]]）

**示例代码**

.. code-block:: python
            
            import paddle.fluid as fluid
            import numpy as np
     
            t = fluid.LoDTensor()
            t.set(np.ndarray([5, 30]), fluid.CPUPlace())
            t.set_lod([[0, 2, 5]])
            print(t.lod()) # [[0, 2, 5]]

.. py:method:: recursive_sequence_lengths(self: paddle.fluid.core_avx.LoDTensor) → List[List[int]]

得到与LoD对应的LoDTensor的序列长度。

返回：LoD对应的一至多个序列长度。

返回类型：out（List [List [int]）

**示例代码**

.. code-block:: python
            
            import paddle.fluid as fluid
            import numpy as np
     
            t = fluid.LoDTensor()
            t.set(np.ndarray([5, 30]), fluid.CPUPlace())
            t.set_recursive_sequence_lengths([[2, 3]])
            print(t.recursive_sequence_lengths()) # [[2, 3]]


.. py:method::  set(*args, **kwargs)
    
重载函数

1. set(self: paddle.fluid.core_avx.Tensor, arg0: numpy.ndarray[float32], arg1: paddle::platform::CPUPlace) -> None

2. set(self: paddle.fluid.core_avx.Tensor, arg0: numpy.ndarray[int32], arg1: paddle::platform::CPUPlace) -> None

3. set(self: paddle.fluid.core_avx.Tensor, arg0: numpy.ndarray[float64], arg1: paddle::platform::CPUPlace) -> None

4. set(self: paddle.fluid.core_avx.Tensor, arg0: numpy.ndarray[int64], arg1: paddle::platform::CPUPlace) -> None

5. set(self: paddle.fluid.core_avx.Tensor, arg0: numpy.ndarray[bool], arg1: paddle::platform::CPUPlace) -> None

6. set(self: paddle.fluid.core_avx.Tensor, arg0: numpy.ndarray[uint16], arg1: paddle::platform::CPUPlace) -> None

7. set(self: paddle.fluid.core_avx.Tensor, arg0: numpy.ndarray[uint8], arg1: paddle::platform::CPUPlace) -> None

8. set(self: paddle.fluid.core_avx.Tensor, arg0: numpy.ndarray[int8], arg1: paddle::platform::CPUPlace) -> None

.. py:method::  set_lod(self: paddle.fluid.core_avx.LoDTensor, lod: List[List[int]]) → None

设置LoDTensor的LoD。

参数：
- **lod** （List [List [int]]） - 要设置的lod。

**示例代码**

.. code-block:: python
            
            import paddle.fluid as fluid
            import numpy as np
     
            t = fluid.LoDTensor()
            t.set(np.ndarray([5, 30]), fluid.CPUPlace())
            t.set_lod([[0, 2, 5]])

.. py:method::  set_recursive_sequence_lengths(self: paddle.fluid.core.LoDTensor, recursive_sequence_lengths: List[List[int]]) → None

根据递归序列长度recursive_sequence_lengths设置LoDTensor的LoD。

例如，如果recursive_sequence_lengths = [[2,3]]，意味着有两个长度分别为2和3的序列，相应的lod将是[[0,2,2 + 3]]，即[[0， 2,5]]。

参数：
- **recursive_sequence_lengths** （List [List [int]]） - 序列长度。

**示例代码**

.. code-block:: python
            
            import paddle.fluid as fluid
            import numpy as np
     
            t = fluid.LoDTensor()
            t.set(np.ndarray([5, 30]), fluid.CPUPlace())
            t.set_recursive_sequence_lengths([[2, 3]])

.. py:method::  shape(self: paddle.fluid.core_avx.Tensor) → List[int]








.. _cn_api_fluid_LoDTensorArray:

LoDTensorArray
-------------------------------

.. py:class:: paddle.fluid.LoDTensorArray

LoDTensor的数组。

**示例代码**

.. code-block:: python
        
        import paddle.fluid as fluid
     
        arr = fluid.LoDTensorArray()   

.. py:method:: append(self: paddle.fluid.core_avx.LoDTensorArray, tensor: paddle.fluid.core.LoDTensor) → None

将LoDTensor追加到LoDTensorArray后。

**示例代码**

.. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
     
            arr = fluid.LoDTensorArray()
            t = fluid.LoDTensor()
            t.set(np.ndarray([5, 30]), fluid.CPUPlace())
            arr.append(t)





.. _cn_api_fluid_memory_optimize:

memory_optimize
-------------------------------

.. py:function:: paddle.fluid.memory_optimize(input_program, skip_opt_set=None, print_log=False, level=0, skip_grads=False)

历史遗留的内存优化策略，通过在不同operators间重用var内存来减少总内存消耗。
用一个简单的示例来解释该算法：

c = a + b  # 假设这里是最后一次使用a
d = b * c

鉴于在“c = a + b”之后不再使用a，且a和d的大小相同，我们可以用变量a来代替变量d，即实际上，上面的代码可以优化成：

c = a + b
a = b * c
     
请注意，在此历史遗存设计中，我们将直接用变量a代替变量d，这意味着在你调用该API后，某些变量将会消失，还有一些会取非预期值。正如上面的例子中，执行程序后，实际上a取d的值。
    
因此，为避免重要变量在优化过程中被重用或移除，我们支持用skip_opt_set指定一个变量白名单。skip_opt_set中的变量不会受memory_optimize API的影响。
     
     
.. note::
    
     此API已被弃用，请不要在你新写的代码中使用它。它不支持block中嵌套子block，如While、IfElse等。

参数:
  - **input_program** (str) – 输入Program。
  - **skip_opt_set** (set) – set中的vars将不被内存优化。
  - **print_log** (bool) – 是否打印debug日志。
  - **level** (int) - 值为0或1。如果level=0，则仅当a.size == b.size时我们才用b代替a；如果level=1，只要a.size <= b.size时我们就可以用b代替a。

返回: None

**示例代码**

.. code-block:: python

    import paddle.fluid as fluid
    main_prog = fluid.Program()
    startup_prog = fluid.Program()
     
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
     
    exe.run(startup_prog)
    fluid.memory_optimize(main_prog)




.. _cn_api_fluid_name_scope:

name_scope
-------------------------------

.. py:function:: paddle.fluid.name_scope(prefix=None)


为operators生成层次名称前缀

注意： 这个函数只能用于调试和可视化。不要将其用于分析，比如graph/program转换。

参数：
  - **prefix** (str) - 前缀

**示例代码**

.. code-block:: python
          
     with fluid.name_scope("s1"):
        a = fluid.layers.data(name='data', shape=[1], dtype='int32')
        b = a + 1
        with fluid.name_scope("s2"):
           c = b * 1
        with fluid.name_scope("s3"):
           d = c / 1
     with fluid.name_scope("s1"):
           f = fluid.layers.pow(d, 2.0)
     with fluid.name_scope("s4"):
           g = f - 1



.. _cn_api_fluid_ParallelExecutor:

ParallelExecutor
-------------------------------

.. py:class:: paddle.fluid.ParallelExecutor(use_cuda, loss_name=None, main_program=None, share_vars_from=None, exec_strategy=None, build_strategy=None, num_trainers=1, trainer_id=0, scope=None)




``ParallelExecutor`` 专门设计用来实现数据并行计算，着力于向不同结点(node)分配数据，并行地在不同结点中对数据进行操作。如果在GPU上使用该类运行程序，node则用来指代GPU， ``ParallelExecutor`` 也将自动获取在当前机器上可用的GPU资源。如果在CPU上进行操作，node则指代CPU，同时你也可以通过添加环境变量 ``CPU_NUM`` 来设置CPU设备的个数。例如，``CPU_NUM=4``。但是如果没有设置该环境变量，该类会调用 ``multiprocessing.cpu_count`` 来获取当前系统中CPU的个数。

**示例代码**

.. code-block:: python

        import paddle.fluid as fluid
        import numpy
        import os
     
        use_cuda = True
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
     
        # 注意：如果你使用CPU运行程序，需要具体设置CPU_NUM，
        # 否则fluid会把逻辑核的所有数目设为CPU_NUM，
        # 在这种情况下，输入的batch size应大于CPU_NUM，
        # 否则程序会异常中断。
        if not use_cuda:
            os.environ['CPU_NUM'] = str(2)
     
        exe = fluid.Executor(place)
     
        train_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            hidden = fluid.layers.fc(input=data, size=10)
            loss = fluid.layers.mean(hidden)
            test_program = fluid.default_main_program().clone(for_test=True)
            fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)
     
        startup_program.random_seed=1
        exe.run(startup_program)
     
        train_exe = fluid.ParallelExecutor(use_cuda=use_cuda,
                                           main_program=train_program,
                                           loss_name=loss.name)
        test_exe = fluid.ParallelExecutor(use_cuda=use_cuda,
                                          main_program=test_program,
                                          share_vars_from=train_exe)
     
        x = numpy.random.random(size=(10, 1)).astype('float32')
        loss_data, = train_exe.run(feed={"X": x},
                                   fetch_list=[loss.name])
     
        loss_data, = test_exe.run(feed={"X": x},
                                  fetch_list=[loss.name])

参数:
    - **use_cuda** (bool) – 是否使用CUDA
    - **loss_name** (str) – 在训练阶段，必须提供loss function名称。默认为None
    - **main_program** (Program) – 需要执行的program。如果未提供， 那么将使用 ``default_main_program``。 默认为None
    - **share_vars_from** (ParallelExecutor) – 如果提供了该参数， 则该 ``ParallelExecutor`` 与指定的 ``ParallelExecutor`` 共享变量。默          认为空
    - **exec_strategy** (ExecutionStrategy) – ``exec_strategy`` 用于调控program在 ``ParallelExecutor`` 中的执行方式，例如，执行该program需要的线程数, 释放在执行过程中产生的临时变量需要的重复(iterations)次数。 请参考 ``fluid.ExecutionStrategy`` 获取详细介绍。该参数默认为 None
    - **build_strategy** (BuildStrategy) – 设置成员 ``build_strategy`` 可以控制在 ``ParallelExecutor`` 中搭建SSA Graph的方式，例如， ``reduce_strategy`` ， ``gradient_scale_strategy`` 。 请参考 ``fluid.BuildStrategy`` 获取详细介绍。 该参数默认为None
    - **num_trainers** (int) – 如果该值大于1， NCCL将会通过多层级node的方式来初始化。每个node应有相同的GPU数目。 随之会启用分布式训练。该参数默认为1
    - **trainer_id** (int) – 必须与 ``num_trainers`` 参数同时使用。``trainer_id`` 是当前所在node的 “rank”（层级），从0开始计数。该参数默认为0
    - **scope** (Scope) – 指定执行program所在的作用域， 默认使用 ``fluid.global_scope()``

返回：初始化后的 ``ParallelExecutor`` 对象

返回类型: ParallelExecutor

抛出异常：``TypeError`` - 如果提供的参数 ``share_vars_from`` 不是 ``ParallelExecutor`` 类型的，将会弹出此异常

.. py:method::  run(fetch_list, feed=None, feed_dict=None, return_numpy=True)

使用 ``fetch_list`` 执行一个 ``ParallelExecutor`` 对象。

参数 ``feed`` 可以是 ``dict`` 或者 ``list`` 类型变量。如果该参数是 ``dict`` 类型，feed中的数据将会被分割(split)并分送给多个设备（CPU/GPU）。
反之，如果它是 ``list`` ，则列表中的各个元素都会直接分别被拷贝到各设备中。

**示例代码**

.. code-block:: python
    
    import paddle.fluid as fluid
    import numpy
    import os

    use_cuda = True
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
     
    # 注意：如果你使用CPU运行程序，需要具体设置CPU_NUM，
    # 否则fluid会把逻辑核的所有数目设为CPU_NUM，
    # 在这种情况下，输入的batch size应大于CPU_NUM，
    # 否则程序会异常中断。
    if not use_cuda:
        os.environ['CPU_NUM'] = str(2)
    exe = fluid.Executor(place)

    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(train_program, startup_program):
        data = fluid.layers.data(name='X', shape=[1], dtype='float32')
        hidden = fluid.layers.fc(input=data, size=10)
        loss = fluid.layers.mean(hidden)
        fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)
 
        startup_program.random_seed=1
        exe.run(startup_program)
 
        train_exe = fluid.ParallelExecutor(use_cuda=use_cuda,
                                           main_program=train_program,
                                           loss_name=loss.name)
    # 如果feed参数是dict类型:
    # 图像会被split到设备中。假设有两个设备，那么每个设备将会处理形为 (5, 1)的图像
    x = numpy.random.random(size=(10, 1)).astype('float32')
    loss_data, = train_exe.run(feed={"X": x},

                               fetch_list=[loss.name])

    # 如果feed参数是list类型:
    # 各设备挨个处理列表中的每个元素
    # 第一个设备处理形为 (10, 1) 的图像
    # 第二个设备处理形为 (9, 1) 的图像
    #
    # 使用 exe.device_count 得到设备数目
    x2 = numpy.random.random(size=(9, 1)).astype('float32')
    loss_data, = train_exe.run(feed=[{"X": x}, {"X": x2}],
                               fetch_list=[loss.name])

参数：
    - **fetch_list** (list) – 获取的变量名列表
    - **feed** (list|dict|None) – feed变量。 如果该参数是 ``dict`` 类型，feed中的数据将会被分割(split)并分送给多个设备（CPU/GPU）。反之，如果它是 ``list`` ，则列表中的各个元素都直接分别被拷贝到各设备中。默认为None
    - **feed_dict** – 该参数已经停止使用。feed参数的别名, 为向后兼容而立。默认为None
    - **return_numpy** (bool) – 是否将fetched tensor转换为numpy。默认为True

返回： 获取的结果列表

返回类型：List

抛出异常:
     - ``ValueError`` - 如果feed参数是list类型，但是它的长度不等于可用设备（执行场所）的数目，再或者给定的feed不是dict类型，抛出此异常
     - ``TypeError`` - 如果feed参数是list类型，但是它里面的元素不是dict类型时，弹出此异常

.. note::
     1. 如果feed参数为dict类型，那么传入 ``ParallelExecutor`` 的数据量 *必须* 大于可用的CPU核数或GPU卡数。否则，C++端将会抛出异常。应额外注意核对数据集的最后一个batch是否比可用的CPU核数或GPU卡数大。
     2. 如果可用的CPU核数或GPU卡数大于一个，则为每个变量最后获取的结果都是list类型，且这个list中的每个元素都是各CPU核或GPU卡上的变量

**代码示例**

.. code-block:: python

        pe = fluid.ParallelExecutor(use_cuda=use_cuda,
                                    loss_name=avg_cost.name,
                                    main_program=fluid.default_main_program())
        loss = pe.run(feed=feeder.feed(cur_batch),
                      fetch_list=[avg_cost.name]))

.. py:method::  drop_local_exe_scopes()

立即删除本地执行作用域。
 
在程序执行期间，生成中间结果被放置在本地执行作用域内，在某些模型中，这些中间结果的创建和删除较为费时。为了解决这个问题，ParallelExecutor在ExecutionStrategy中提供了可选项，如num_iteration_per_drop_scope，此选项指示在删除本地执行作用域之前要运行的迭代次数。 但在某些情况下，每次迭代都会产生不同的中间结果，这将导致本地执行作用域所需的内存逐渐增加。 如果你想在这个时候运行另一个程序，可能没有足够的存储空间，此时你应该删除其他程序的本地执行作用域。
     

**代码示例**

.. code-block:: python

            import paddle.fluid as fluid
            import numpy
            import os
     
            use_cuda = True
            # 注意：如果你使用CPU运行程序，需要具体设置CPU_NUM，
            # 否则fluid会把逻辑核的所有数目设为CPU_NUM，
            # 在这种情况下，输入的batch size应大于CPU_NUM，
            # 否则程序会异常中断。
            if not use_cuda:
                os.environ['CPU_NUM'] = str(2)
     
            train_program = fluid.Program()
            startup_program = fluid.Program()
            with fluid.program_guard(train_program, startup_program):
                data = fluid.layers.data(name='X', shape=[1], dtype='float32')
                hidden = fluid.layers.fc(input=data, size=10)
                loss = fluid.layers.mean(hidden)
     
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe.run(startup_program)
     
            parallel_exe = fluid.ParallelExecutor(use_cuda=use_cuda,
                                               main_program=train_program,
                                               loss_name=loss.name)
     
            x = numpy.random.random(size=(10, 1)).astype('float32')
            loss_data, = parallel_exe.run(feed={"X": x},
                                       fetch_list=[loss.name])
     
            parallel_exe.drop_local_exe_scopes()




.. _cn_api_fluid_ParamAttr:


ParamAttr
-------------------------------


.. py:class:: paddle.fluid.ParamAttr(name=None, initializer=None, learning_rate=1.0, regularizer=None, trainable=True, gradient_clip=None, do_model_average=False)

该类代表了参数的各种属性。 为了使神经网络训练过程更加流畅，用户可以根据需要调整参数属性。比如learning rate（学习率）, regularization（正则化）, trainable（可训练性）, do_model_average(平均化模型)和参数初始化方法.

参数:
    - **name** (str) – 参数名。默认为None。
    - **initializer** (Initializer) – 初始化该参数的方法。 默认为None
    - **learning_rate** (float) – 参数的学习率。计算方法为 :math:`global\_lr*parameter\_lr∗scheduler\_factor` 。 默认为1.0
    - **regularizer** (WeightDecayRegularizer) – 正则因子. 默认为None
    - **trainable** (bool) – 该参数是否可训练。默认为True
    - **gradient_clip** (BaseGradientClipAttr) – 减少参数梯度的方法。默认为None
    - **do_model_average** (bool) – 该参数是否服从模型平均值。默认为False

**代码示例**

.. code-block:: python

   import paddle.fluid as fluid
   
   w_param_attrs = fluid.ParamAttr(name="fc_weight",
                                   learning_rate=0.5,
                                   regularizer=fluid.L2Decay(1.0),
                                   trainable=True)
   y_predict = fluid.layers.fc(input=x, size=10, param_attr=w_param_attrs)













.. _cn_api_fluid_Program:

Program
-------------------------------

.. py:class::  paddle.fluid.Program


创建python program， 在paddleFluid内部会被转换为ProgramDesc描述语言，用来创建一段 c++ 程序。Program像容器一样，是一种自包含的程序语言。Program中包括至少一个块（Block），当 block 中存在条件选择的控制流op（例如 while_op）时，该Program将会含有嵌套块（nested block）。详情请参阅framework.proto。

注意：默认情况下，paddleFluid内部默认含有 ``default_startup_program`` 和 ``default_main_program`` ，它们将共享参数。 ``default_startup_program`` 只运行一次来初始化参数， ``default_main_program`` 在每个mini batch中运行并调整权重。

返回： empty program

**代码示例**

.. code-block:: python
  
    import paddle.fluid as fluid

    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(main_program=main_program, startup_program=startup_program):
        x = fluid.layers.data(name="x", shape=[-1, 784], dtype='float32')
        y = fluid.layers.data(name="y", shape=[-1, 1], dtype='int32')
        z = fluid.layers.fc(name="fc", input=x, size=10, act="relu")

    print("main program is: {}".format(main_program))
      
    print("start up program is: {}".format(startup_program))


.. py:method:: to_string(throw_on_error, with_details=False)

用于debug

参数：
  - **throw_on_error** (bool): 没有设置任何必需的字段时，抛出值错误。
  - **with_details** (bool): 值为true时，打印更多关于变量和参数的信息，如trainable, optimize_attr等

返回：(str): debug 字符串

返回类型： str

抛出异常：
 - ``ValueError`` - 当 ``throw_on_error == true`` ，但没有设置任何必需的字段时，抛出 ``ValueError`` 。

**代码示例**

.. code-block:: python

            import paddle.fluid as fluid
     
            prog = fluid.default_main_program()
            prog_string = prog.to_string(throw_on_error=True, with_details=False)
            print(prog_string)

.. py:method:: clone(for_test=False)

创建一个新的、相同的Program。

有些operator，在训练和测试之间的行为是不同的，比如 ``batch_norm`` 。它们有一个属性 ``is_test`` 来控制行为。当 ``for_test=True`` 时，此方法将把它们的 ``is_test`` 属性更改为True。

- 克隆Program用于训练时，将 ``for_test`` 设置为False。
- 克隆Program用于测试时，将 ``for_test`` 设置为True。我们不会在此处对程序进行任何裁剪，因此，如果您只是想要一个用于测试的前向计算程序，请在使用 ``Opimizer.minimize`` 之前使用 ``clone``

注意: 
    1. ``Program.clone()`` 方法不会克隆 ``py_reader`` 
    2. 此API不会裁剪任何算子。请在backward和optimization之前使用 ``clone(for_test=True)`` 。例如：

    .. code-block:: python

          test_program = fluid.default_main_program().clone(for_test=True)
          optimizer = fluid.optimizer.Momentum(learning_rate=0.01, momentum=0.9)
          optimizer.minimize()

参数：
  - **for_test** (bool) – 取值为True时，clone方法内部会把operator的属性 ``is_test`` 设置为 True

返回：一个新的、相同的Program

返回类型：Program

**代码示例**

注意，Program Desc在clone后的顺序可能不同，这不会影响您的训练或测试进程。在下面的示例中，我们为您提供了一个简单的方法print_prog（program）来打印程序描述，以确保clone后您仍能得到同样的打印结果：

.. code-block:: python     
                
        import paddle.fluid as fluid
        import six


        def print_prog(prog):
            for name, value in sorted(six.iteritems(prog.block(0).vars)):
                print(value)
            for op in prog.block(0).ops:
                print("op type is {}".format(op.type))
                print("op inputs are {}".format(op.input_arg_names))
                print("op outputs are {}".format(op.output_arg_names))
                for key, value in sorted(six.iteritems(op.all_attrs())):
                    if key not in ['op_callstack', 'op_role_var']:
                        print(" [ attrs: {}:   {} ]".format(key, value))

1.克隆一个Program，示例代码如下。

.. code-block:: python

        import paddle.fluid as fluid
        import six

        def print_prog(prog):
            for name, value in sorted(six.iteritems(prog.block(0).vars)):
                print(value)
            for op in prog.block(0).ops:
                print("op type is {}".format(op.type))
                print("op inputs are {}".format(op.input_arg_names))
                print("op outputs are {}".format(op.output_arg_names))
                for key, value in sorted(six.iteritems(op.all_attrs())):
                    if key not in ['op_callstack', 'op_role_var']:
                        print(" [ attrs: {}:   {} ]".format(key, value))

        train_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            with fluid.unique_name.guard():
                img = fluid.layers.data(name='image', shape=[784])
                hidden = fluid.layers.fc(input=img, size=200, act='relu')
                hidden = fluid.layers.dropout(hidden, dropout_prob=0.5)
                loss = fluid.layers.cross_entropy(
                                          input=fluid.layers.fc(hidden, size=10, act='softmax'),
                            label=fluid.layers.data(name='label', shape=[1], dtype='int64'))
                avg_loss = fluid.layers.mean(loss)
                test_program = train_program.clone(for_test=False)
        print_prog(test_program)
        with fluid.program_guard(train_program, startup_program):
            with fluid.unique_name.guard():
                sgd = fluid.optimizer.SGD(learning_rate=1e-3)
                sgd.minimize(avg_loss)
  
2.如果分别运行 train Program 和 test Program，则可以不使用clone。

.. code-block:: python

        import paddle.fluid as fluid
        import six

        def print_prog(prog):
            for name, value in sorted(six.iteritems(prog.block(0).vars)):
                print(value)
            for op in prog.block(0).ops:
                print("op type is {}".format(op.type))
                print("op inputs are {}".format(op.input_arg_names))
                print("op outputs are {}".format(op.output_arg_names))
                for key, value in sorted(six.iteritems(op.all_attrs())):
                    if key not in ['op_callstack', 'op_role_var']:
                        print(" [ attrs: {}:   {} ]".format(key, value))
        def network(is_test):
            img = fluid.layers.data(name='image', shape=[784])
            hidden = fluid.layers.fc(input=img, size=200, act='relu')
            hidden = fluid.layers.dropout(hidden, dropout_prob=0.5)
            loss = fluid.layers.cross_entropy(
                input=fluid.layers.fc(hidden, size=10, act='softmax'),
                label=fluid.layers.data(name='label', shape=[1], dtype='int64'))
            avg_loss = fluid.layers.mean(loss)
            return avg_loss


        train_program_2 = fluid.Program()
        startup_program_2 = fluid.Program()
        test_program_2 = fluid.Program()
        with fluid.program_guard(train_program_2, startup_program_2):
            with fluid.unique_name.guard():
                 sgd = fluid.optimizer.SGD(learning_rate=1e-3)
                 sgd.minimize(avg_loss)
        # 不使用测试阶段的启动程序
        with fluid.program_guard(test_program_2, fluid.Program()):
            with fluid.unique_name.guard():
                loss = network(is_test=True)
        print(test_program_2)

上边两个代码片段生成和打印的Program是一样的。

.. py:staticmethod:: parse_from_string(binary_str)

反序列化protobuf，转换成program

注意:在序列化和反序列化之后，所有关于参数的信息都会丢失。

参数:
    - **binary_str_type** (str) – prootbuf二进制字符串

返回: 反序列化后的ProgramDesc

返回类型：Program

.. py:attribute:: num_blocks

该program中的block的个数

**代码示例**

.. code-block:: python
            
            import paddle.fluid as fluid
     
            prog = fluid.default_main_program()
            num_blocks = prog.num_blocks
            print(num_blocks)

.. py:attribute:: random_seed


程序中随机运算符的默认随机种子。0意味着从随机设备中获取随机种子。

注意：必须在operator被添加之前设置。

**代码示例**

.. code-block:: python

            import paddle.fluid as fluid
     
            prog = fluid.default_main_program()
            random_seed = prog.random_seed
            print(random_seed)
            prog.random_seed = 1
            print(prog.random_seed)

.. py:method:: global_block()

获取该program的第一个block。

**代码示例**

.. code-block:: python

            import paddle.fluid as fluid
     
            prog = fluid.default_main_program()
            gb_block = prog.global_block()
            print(gb_block)

.. py:method:: block(index)

返回该program中 ， ``index`` 指定的block。 ``index`` 类型为int

返回：index对应的block

返回类型：Block

**代码示例**

.. code-block:: python

            import paddle.fluid as fluid
     
            prog = fluid.default_main_program()
            block_0 = prog.block(0)
            print(block_0)

.. py:method:: current_block()

获取当前block。当前block是用来添加operators。

**代码示例**

.. code-block:: python

            import paddle.fluid as fluid
     
            prog = fluid.default_main_program()
            current_blk = prog.current_block()
            print(current_blk)

.. py:method:: list_vars()

获取当前program中所有变量。返回值是一个可迭代对象（iterable object)。

返回：generator 会yield每个Program中的变量

返回类型：iterable
  
**代码示例**

.. code-block:: python

            import paddle.fluid as fluid
     
            prog = fluid.default_main_program()
            img = fluid.layers.data(name='img', shape=[1,28,28], dtype='float32')
            label = fluid.layers.data(name='label', shape=[128,1], dtype='int64')
            for var in prog.list_vars():
                print(var)




.. _cn_api_fluid_program_guard:

program_guard
-------------------------------

.. py:function::    paddle.fluid.program_guard(main_program, startup_program=None)



该函数应配合使用python的“with”语句来改变全局主程序(main program)和启动程序(startup program)。

“with”语句块中的layer函数将在新的main program（主程序）中添加operators（算子）和variables（变量）。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(main_program, startup_program):
        data = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')
        hidden = fluid.layers.fc(input=data, size=10, act='relu')

需要注意的是，如果用户不需要构建自己的启动程序或者主程序，一个临时的program将会发挥作用。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    main_program = fluid.Program()
    # 如果您不需要关心startup program,传入一个临时值即可
    with fluid.program_guard(main_program, fluid.Program()):
        data = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')


参数：
    - **main_program** (Program) – “with”语句中将使用的新的main program。
    - **startup_program** (Program) – “with”语句中将使用的新的startup program。若传入 ``None`` 则不改变当前的启动程序。










.. _cn_api_fluid_release_memory:

release_memory
-------------------------------

.. py:function:: paddle.fluid.release_memory(input_program, skip_opt_set=None)


该函数可以调整输入program，插入 ``delete_op`` 删除算子，提前删除不需要的变量。
改动是在变量本身上进行的。

**提醒**: 该API还在试验阶段，会在后期版本中删除。不建议用户使用。

参数:
    - **input_program** (Program) – 在此program中插入 ``delete_op``
    - **skip_opt_set** (set) – 在内存优化时跳过的变量的集合

返回: None

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
     
    # 搭建网络
    # ...
     
    # 已弃用的API
    fluid.release_memory(fluid.default_main_program())
     



.. _cn_api_fluid_scope_guard:

scope_guard
-------------------------------

.. py:function:: paddle.fluid.scope_guard(scope)


修改全局/默认作用域（scope）,  运行时中的所有变量都将分配给新的scope。

参数：
  - **scope** - 新的全局/默认 scope。

**代码示例**

.. code-block:: python

  import numpy
  
  new_scope = fluid.Scope()
  with fluid.scope_guard(new_scope):
       fluid.global_scope().var("data").get_tensor().set(numpy.ones((2, 2)), fluid.CPUPlace())
  numpy.array(new_scope.find_var("data").get_tensor())
 




.. _cn_api_fluid_Tensor:

Tensor
-------------------------------

.. py:function:: paddle.fluid.Tensor

    ``LoDTensor`` 的别名









.. _cn_api_fluid_WeightNormParamAttr:

WeightNormParamAttr
-------------------------------

.. py:class:: paddle.fluid.WeightNormParamAttr(dim=None, name=None, initializer=None, learning_rate=1.0, regularizer=None, trainable=True, gradient_clip=None, do_model_average=False)


权重归一化。权重归一化是将权重向量的长度与其方向解耦。`Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks <https://arxiv.org/pdf/1602.07868.pdf>`_ 这篇paper中讨论了权重归一化的实现

参数:
  - **dim** (list) - 参数的名称。默认None。
  - **name** (str) - 参数的名称。默认None。
  - **initializer** （initializer) - 初始化参数的方法。默认None。
  - **learning_rate** (float) - 学习率。优化时学习速率 :math:`global\_lr∗parameter\_lr∗scheduler\_factor` 。默认1.0。
  - **regularizer** (WeightDecayRegularizer) - 正则化因子。默认None。
  - **trainable** (bool) - 参数是否可训练。默认True。
  - **gradient_clip** (BaseGradientClipAttr) - 梯度下降裁剪（Gradient Clipping）的方法。默认None。
  - **do_model_average** (bool) - 参数是否应该model average。默认False。

返回： empty program

**代码示例**

.. code-block:: python
      
  import paddle.fluid as fluid
  data = fluid.layers.data(name="data", shape=[3, 32, 32], dtype="float32")
  fc = fluid.layers.fc(input=data,
                       size=1000,
                       param_attr=fluid.WeightNormParamAttr(
                                dim=None,
                                name='weight_norm_param'))








