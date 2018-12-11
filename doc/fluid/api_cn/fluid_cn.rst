#################
 fluid
#################



.. _cn_api_fluid_AsyncExecutor:

AsyncExecutor
-------------------------------

.. py:function:: paddle.fluid.AsyncExecutor(place=None)

Python中的异步执行器。AsyncExecutor利用多核处理器和数据排队的强大功能，使数据读取和融合解耦，每个线程并行运行。

AsyncExecutor不是在python端读取数据，而是接受一个训练文件列表，该列表将在c++中检索，然后训练输入将被读取、解析并在c++代码中提供给训练网络。

AsyncExecutor正在积极开发，API可能在不久的将来会发生变化。

参数：
	- **place** (fluid.CPUPlace|None) - 指示 executor 将在哪个设备上运行。目前仅支持CPU

**代码示例：**

.. code-block:: python

    data_feed = fluid.DataFeedDesc('data.proto')
    startup_program = fluid.default_startup_program()
    main_program = fluid.default_main_program()
    filelist = ["train_data/part-%d" % i for i in range(100)]
    thread_num = len(filelist) / 4
    place = fluid.CPUPlace()
    async_executor = fluid.AsyncExecutor(place)
    async_executor.run_startup_program(startup_program)
    epoch = 10
    for i in range(epoch):
        async_executor.run(main_program,
                           data_feed,
                           filelist,
                           thread_num,
                           [acc],
                           debug=False)

.. note::

	对于并行gpu调试复杂网络，您可以在executor上测试。他们有完全相同的参数，并可以得到相同的结果。

	目前仅支持CPU

.. py:method:: run(program, data_feed, filelist, thread_num, fetch, debug=False)

使用此 ``AsyncExecutor`` 来运行 ``program`` 。

``filelist`` 中包含训练数据集。用户也可以通过在参数 ``fetch`` 中提出变量来检查特定的变量， 正如 ``fluid.Executor`` 。

但不像 ``fluid.Executor`` ， ``AsyncExecutor`` 不返回获取到的变量，而是将每个获取到的变量作为标准输出展示给用户。

数据集上的运算在多个线程上执行，每个线程中都会独立出一个线程本地作用域，并在此域中建立运算。
所有运算同时更新参数值。

参数:	
  - program (Program) – 需要执行的program。如果没有提供该参数，默认使用 ``default_main_program`` 
  - data_feed (DataFeedDesc) –  ``DataFeedDesc`` 对象
  - filelist (str) – 一个包含训练数据集文件的文件列表
  - thread_num (int) – 并发训练线程数。参照 *注解* 部分获取合适的设置方法
  - fetch (str|list) – 变量名，或者变量名列表。指明最后要进行观察的变量命名
  - debug (bool) – 如果为True, 在每一个minibatch处理后，fetch 中指明的变量将会通过标准输出打印出来

.. note::
    1.该执行器会运行program中的所有运算，不只是那些依赖于fetchlist的运算

    2.该类执行器在多线程上运行，每个线程占用一个CPU核。为了实现效率最大化，建议将 ``thread_num`` 等于或稍微小于CPU核心数






.. _cn_api_fluid_BuildStrategy:

BuildStrategy
-------------------------------

.. py:class::  paddle.fluid.BuildStrategy

``BuildStrategy`` 使用户更精准地控制 ``ParallelExecutor`` 中SSA图的建造方法。可通过设置 ``ParallelExecutor`` 中的 ``BuildStrategy`` 成员来实现此功能。

**代码示例**

..  code-block:: python

    build_strategy = fluid.BuildStrategy()
    build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce

    train_exe = fluid.ParallelExecutor(use_cuda=True,
                                       loss_name=loss.name,
                                       build_strategy=build_strategy)

    train_loss, = train_exe.run([loss.name], feed=feed_dict)



.. py:attribute:: debug_graphviz_path

str类型。它表明了以graphviz格式向文件中写入SSA图的路径，有利于调试。 默认值为""。



.. py:attribute:: fuse_elewise_add_act_ops

bool类型。它表明了是否融合（fuse）elementwise_add_op和activation_op。这会使整体执行过程更快一些。默认为False。



.. py:attribute:: gradient_scale_strategy

str类型。在 ``ParallelExecutor`` 中，存在三种定义 *loss@grad* 的方式，分别为 ``CoeffNumDevice``, ``One`` 与 ``Customized``。默认情况下， ``ParallelExecutor`` 根据设备数目来设置 *loss@grad* 。如果你想自定义 *loss@grad* ，你可以选择 ``Customized`` 方法。默认为 ``CoeffNumDevice`` 。



.. py:attribute:: reduce_strategy

str类型。在 ``ParallelExecutor`` 中，存在两种减少策略（reduce strategy），即 ``AllReduce`` 和 ``Reduce`` 。如果你需要在所有执行场所上独立地进行参数优化，可以使用 ``AllReduce`` 。反之，如果使用 ``Reduce`` 策略，所有参数的优化将均匀地分配给不同的执行场所，随之将优化后的参数广播给其他执行场所。在一些模型中， ``Reduce`` 策略执行速度更快一些。默认值为 ``AllReduce`` 。












.. _cn_api_fluid_CPUPlace:

CPUPlace
-------------------------------

.. py:class:: paddle.fluid.CPUPlace








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
         假如我们想用LoD Tensor来承载一词序列的数据，其中每个词由一个整数来表示。现在，我们意图创建一个LoD Tensor来代表两个句子，其中一个句子有两个词，另外一个句子有三个。
     	 那么数 ``data`` 可以是一个numpy数组，形状为（5,1）。同时， ``recursive_seq_lens`` 为 [[2, 3]]，表明各个句子的长度。这个长度为基准的 ``recursive_seq_lens`` 将在函数中会被转化为以偏移量为基准的 LoD [[0, 2, 5]]。

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

返回:	一个fluid LoDTensor对象，包含数据和 ``recursive_seq_lens`` 信息













.. _cn_api_fluid_CUDAPinnedPlace:

CUDAPinnedPlace
-------------------------------

.. py:class:: paddle.fluid.CUDAPinnedPlace












.. _cn_api_fluid_CUDAPlace:

CUDAPlace
-------------------------------

.. py:class:: paddle.fluid.CUDAPlace








.. _cn_api_fluid_DataFeedDesc:

DataFeedDesc
-------------------------------

.. py:function:: paddle.fluid.DataFeedDesc(proto_file)

数据描述符，描述输入训练数据格式。

这个类目前只用于AsyncExecutor(有关类AsyncExecutor的简要介绍，请参阅注释)

DataFeedDesc应由来自磁盘的有效protobuf消息初始化:

.. code-block:: python

	data_feed = fluid.DataFeedDesc('data.proto')

可以参考 :code:`paddle/fluid/framework/data_feed.proto` 查看我们如何定义message

一段典型的message可能是这样的：

.. code-block:: text

    name: "MultiSlotDataFeed"
    batch_size: 2
    multi_slot_desc {
        slots {
            name: "words"
            type: "uint64"
            is_dense: false
            is_used: true
        }
        slots {
            name: "label"
            type: "uint64"
            is_dense: false
            is_used: true
        }
    }

但是，用户通常不应该关心消息格式;相反，我们鼓励他们在将原始日志文件转换为AsyncExecutor可以接受的训练文件的过程中，使用 :code:`Data Generator` 生成有效数据描述。

DataFeedDesc也可以在运行时更改。一旦你熟悉了每个字段的含义，您可以修改它以更好地满足您的需要。例如:

.. code-block:: python

    data_feed.set_batch_size(128)
    data_feed.set_dense_slots('wd')  # The slot named 'wd' will be dense
    data_feed.set_use_slots('wd')    # The slot named 'wd' will be used
    
    #Finally, the content can be dumped out for debugging purpose:
    
    print(data_feed.desc())


参数：
	- **proto_file** (string) - 包含数据feed中描述的磁盘文件


.. py:method:: set_batch_size(self, batch_size)

设置batch size，训练期间有效


参数：
	- batch_size：batch size

**代码示例：**

.. code-block:: python
	
	data_feed = fluid.DataFeedDesc('data.proto')
	data_feed.set_batch_size(128)

.. py:method:: set_dense_slots(self, dense_slots_name)

指定slot经过设置后将变成密集的slot，仅在训练期间有效。

密集slot的特征将被输入一个Tensor，而稀疏slot的特征将被输入一个lodTensor


参数：
	- **dense_slots_name** : slot名称的列表，这些slot将被设置为密集的

**代码示例：**

.. code-block:: python
	
	data_feed = fluid.DataFeedDesc('data.proto')
	data_feed.set_dense_slots(['words'])

.. note:: 

	默认情况下，所有slot都是稀疏的

.. py:method:: set_use_slots(self, use_slots_name)


设置一个特定的slot是否用于训练。一个数据集包含了很多特征，通过这个函数可以选择哪些特征将用于指定的模型。

参数：
	- **use_slots_name** :将在训练中使用的slot名列表

**代码示例：**

.. code-block:: python

	data_feed = fluid.DataFeedDesc('data.proto')
	data_feed.set_use_slots(['words'])

.. note::
	
	默认值不用于所有slot


.. py:method:: desc(self)

返回此DataFeedDesc的protobuf信息

返回：一个message字符串

**代码示例：**

.. code-block:: python

	data_feed = fluid.DataFeedDesc('data.proto')
	print(data_feed.desc())






.. _cn_api_fluid_DataFeeder:

DataFeeder
-------------------------------

.. py:class:: paddle.fluid.DataFeeder(feed_list, place, program=None)



``DataFeeder`` 负责将reader(读取器)返回的数据转成一种特殊的数据结构，使它们可以输入到 ``Executor`` 和 ``ParallelExecutor`` 中。
reader通常返回一个minibatch条目列表。在列表中每一条目都是一个样本（sample）,它是由具有一至多个特征的列表或元组组成的。


以下是简单用法：

..  code-block:: python
	
	place = fluid.CPUPlace()
	img = fluid.layers.data(name='image', shape=[1, 28, 28])
	label = fluid.layers.data(name='label', shape=[1], dtype='int64')
	feeder = fluid.DataFeeder([img, label], fluid.CPUPlace())
	result = feeder.feed([([0] * 784, [9]), ([1] * 784, [1])])
	
在多GPU模型训练时，如果需要提前分别向各GPU输入数据，可以使用 ``decorate_reader`` 函数。

..  code-block:: python

	place=fluid.CUDAPlace(0)
	feeder = fluid.DataFeeder(place=place, feed_list=[data, label])
	reader = feeder.decorate_reader(
    		paddle.batch(flowers.train(), batch_size=16))



参数：
    - **feed_list** (list) – 向模型输入的变量表或者变量表名
    - **place** (Place) – place表明是向GPU还是CPU中输入数据。如果想向GPU中输入数据, 请使用 ``fluid.CUDAPlace(i)`` (i 代表 the GPU id)；如果向CPU中输入数据, 请使用  ``fluid.CPUPlace()``
    - **program** (Program) – 需要向其中输入数据的Program。如果为None, 会默认使用 ``default_main_program()``。 缺省值为None


抛出异常:
  - ``ValueError``  – 如果一些变量不在此 Program 中


**代码示例**

..  code-block:: python

	# ...
	place = fluid.CPUPlace()
	feed_list = [
    		main_program.global_block().var(var_name) for var_name in feed_vars_name
	] # feed_vars_name 是一个由变量名组成的列表
	feeder = fluid.DataFeeder(feed_list, place)
	for data in reader():
    		outs = exe.run(program=main_program,
               		       feed=feeder.feed(data))
			       
			       
.. py:method:: feed(iterable)


根据feed_list（数据输入表）和iterable（可遍历的数据）提供的信息，将输入数据转成一种特殊的数据结构，使它们可以输入到 ``Executor`` 和 ``ParallelExecutor`` 中。

参数:	
	- **iterable** (list|tuple) – 要输入的数据

返回：  转换结果

返回类型:	dict


.. py:method:: feed_parallel(iterable, num_places=None)


该方法获取的多个minibatch，并把每个minibatch提前输入进各个设备中。

参数:	
    - **iterable** (list|tuple) – 要输入的数据
    - **num_places** (int) – 设备数目。默认为None。

返回: 转换结果

返回类型: dict

.. note::
     设备（CPU或GPU）的数目必须等于minibatch的数目



.. py:method::  decorate_reader(reader, multi_devices, num_places=None, drop_last=True)


  
将reader返回的输入数据batch转换为多个mini-batch，之后每个mini-batch都会被输入进各个设备（CPU或GPU）中。
    
参数：
        - **reader** (fun) – 该参数是一个可以生成数据的函数
        - **multi_devices** (bool) – bool型，指明是否使用多个设备
        - **num_places** (int) – 如果 ``multi_devices`` 为 ``True`` , 可以使用此参数来设置GPU数目。如果 ``num_places`` 为 ``None`` ，该函数默认使用当前训练机所有GPU设备。默认为None。
        - **drop_last** (bool) – 如果最后一个batch的大小比 ``batch_size`` 要小，则可使用该参数来指明是否选择丢弃最后一个batch数据。 默认为 ``True`` 

返回：转换结果

返回类型: dict
    
抛出异常： ``ValueError`` – 如果 ``drop_last`` 值为False并且reader返回的minibatch数目与设备数目不相等时，产生此异常


        









.. _cn_api_fluid_default_main_program:

default_main_program
-------------------------------

.. py:function:: paddle.fluid.default_main_program()





此函数用于获取默认或全局main program(主程序)。该主程序用于训练和测试模型。

``fluid.layers`` 中的所有layer函数可以向 ``default_main_program`` 中添加operators（算子）和variables（变量）。

``default_main_program`` 是fluid的许多编程接口（API）的Program参数的缺省值。例如,当用户program没有传入的时候，
``Executor.run()`` 会默认执行 ``default_main_program`` 。


返回：	main program

返回类型:	Program











.. _cn_api_fluid_default_startup_program:




default_startup_program
-------------------------------

.. py:function:: paddle.fluid.default_startup_program()



该函数可以获取默认/全局 startup program (启动程序)。

``fluid.layers`` 中的layer函数会新建参数、readers(读取器)、NCCL句柄作为全局变量。 

startup_program会使用内在的operators（算子）去初始化他们，并由layer函数将这些operators追加到startup program中。

该函数将返回默认的或当前的startup_program。用户可以使用 ``fluid.program_guard`` 去切换program。

返回:	startup program

返回类型:	Program











.. _cn_api_fluid_DistributeTranspiler:

DistributeTranspiler
-------------------------------

.. py:class:: paddle.fluid.DistributeTranspiler (config=None)


该类可以把fluid program转变为分布式数据并行计算程序（distributed data-parallelism programs）,可以有Pserver和NCCL2两种模式。
当program在Pserver（全称：parameter server）模式下， ``main_program`` (主程序)转为使用一架远程parameter server(即pserver,参数服务器)来进行参数优化，并且优化图会被输入到一个pserver program中。
在NCCL2模式下，transpiler会在 ``startup_program`` 中附加一个 ``NCCL_ID`` 广播算子（broadcasting operators）来实现在该集群中所有工作结点共享 ``NCCL_ID`` 。
调用 ``transpile_nccl2`` 后， 你 **必须** 将 ``trainer_id`` , ``num_trainers`` 参数提供给 ``ParallelExecutor`` 来启动NCCL2分布式模式。 




**代码示例**

..  code-block:: python

	#pserver模式下
	pserver_endpoints = "192.168.0.1:6174,192.168.0.2:6174"
	trainer_endpoints = "192.168.0.1:6174,192.168.0.2:6174"
	current_endpoint = "192.168.0.1:6174"
	trainer_id = 0
	trainers = 4
	role = os.getenv("PADDLE_TRAINING_ROLE")

	t = fluid.DistributeTranspiler()
	t.transpile(
     	     trainer_id, pservers=pserver_endpoints, trainers=trainers)
	if role == "PSERVER":
     	     pserver_program = t.get_pserver_program(current_endpoint)
             pserver_startup_program = t.get_startup_program(current_endpoint,
                                                     pserver_program)
	elif role == "TRAINER":
             trainer_program = t.get_trainer_program()

	# nccl2模式下
	config = fluid.DistributeTranspilerConfig()
	config.mode = "nccl2"
	t = fluid.DistributeTranspiler(config=config)
	t.transpile(trainer_id, workers=workers, current_endpoint=curr_ep)
	exe = fluid.ParallelExecutor(
    	    use_cuda,
            loss_name=loss_var.name,
            num_trainers=len(trainers.split(",)),
            trainer_id=trainer_id
	)



.. py:method:: transpile(trainer_id, program=None, pservers='127.0.0.1:6174', trainers=1, sync_mode=True, startup_program=None, current_endpoint='127.0.0.1:6174')

该方法可以运行该transpiler（转译器）。

参数:	
	- **trainer_id** (int) – 当前Trainer worker的id, 如果有n个Trainer worker, id 取值范围为0 ~ n-1
	- **program** (Program|None) – 待transpile（转译）的program, 缺省为 ``fluid.default_main_program()`` 
	- **startup_program** (Program|None) - 要转译的 ``startup_program`` ,默认为 ``fluid.default_startup_program()``
	- **pservers** (str) – 内容为Pserver列表的字符串，格式为：按逗号区分不同的Pserver，每个Pserver的格式为 *ip地址:端口号* 
	- **trainers** (int|str) – 在Pserver模式下，该参数指Trainer机的个数；在nccl2模式下，它是一个内容为Trainer终端列表的字符串
	- **sync_mode** (bool) – 是否做同步训练(synchronous training), 默认为True
 	- **startup_program** (Program|None) – 待transpile（转译）的startup_program，默认为 ``fluid.default_main_program()``
	- **current_endpoint** (str) – 当需要把program转译（transpile）至NCCL2模式下时，需要将当前endpoint（终端）传入该参数。Pserver模式不使用该参数

.. py:method:: get_trainer_program(wait_port=True)


该方法可以得到Trainer侧的program。

返回:	Trainer侧的program

返回类型:	Program



.. py:method:: get_pserver_program(endpoint)


该方法可以得到Pserver（参数服务器）侧的程序
 
参数:	
	- **endpoint** (str) – 当前Pserver终端
 
返回:	当前Pserver需要执行的program

返回类型:	Program


.. py:method:: get_pserver_programs(endpoint)


该方法可以得到Pserver侧用于分布式训练的 ``main_program`` 和 ``startup_program`` 。

参数:	
	- **endpoint** (str) – 当前Pserver终端

返回:	(main_program, startup_program), “Program”类型的元组

返回类型:	tuple 
 
 
.. py:method:: get_startup_program(endpoint, pserver_program=None, startup_program=None)


**该函数已停止使用**
获取当前Pserver的startup_program，如果有多个被分散到不同blocks的变量，则修改operator的输入变量。

参数:	
	- **endpoint** (str) – 当前Pserver终端
	- **pserver_program** (Program) – 已停止使用。 先调用get_pserver_program
 	- **startup_program** (Program) – 已停止使用。应在初始化时传入startup_program

返回:	Pserver侧的startup_program

返回类型:	Program









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

最小数据块的大小

注意: 根据：`issuecomment-369912156 <https://github.com/PaddlePaddle/Paddle/issues/8638#issuecomment-369912156>`_ , 当数据块大小超过2MB时，我们可以有效地使用带宽。如果你想更改它，请详细查看 ``slice_variable`` 函数。







.. _cn_api_fluid_ExecutionStrategy:

ExecutionStrategy
-------------------------------

.. py:class:: paddle.fluid.ExecutionStrategy

``ExecutionStrategy`` 允许用户更加精准地控制program在 ``ParallelExecutor`` 中的运行方式。可以通过在 ``ParallelExecutor`` 中设置本成员来实现。

**代码示例**

..  code-block:: python

  exec_strategy = fluid.ExecutionStrategy()
  exec_strategy.num_threads = 4

  train_exe = fluid.ParallelExecutor(use_cuda=True,
                                     loss_name=loss.name,
                                     exec_strategy=exec_strategy)

  train_loss, = train_exe.run([loss.name], feed=feed_dict)



.. py:attribute:: allow_op_delay
   
这是一个bool类型成员，表示是否推迟communication operators(交流运算)的执行，这样做会使整体执行过程更快一些。但是在一些模型中，allow_op_delay会导致程序中断。默认为False。
  


.. py:attribute:: num_iteration_per_drop_scope
  
int型成员。它表明了清空执行时产生的临时变量需要的程序执行重复次数。因为临时变量的形可能在两次重复过程中保持一致，所以它会使整体执行过程更快。默认值为100。

.. note::
  1. 如果在调用 ``run`` 方法时获取结果数据，``ParallelExecutor`` 会在当前程序重复执行尾部清空临时变量
  
  2. 在一些NLP模型里，该成员会致使GPU内存不足。此时，你应减少 ``num_iteration_per_drop_scope`` 的值



.. py:attribute:: num_threads

int型成员。它代表了线程池(thread pool)的大小。这些线程会被用来执行当前 ``ParallelExecutor`` 的program中的operator（算子，运算）。如果 :math:`num\_threads=1` ，则所有的operator将一个接一个地执行，但在不同的程序重复周期(iterations)中执行顺序可能不同。如果该成员没有被设置，则在 ``ParallelExecutor`` 中，它会依据设备类型(device type)、设备数目(device count)而设置为相应值。对GPU，:math:`num\_threads=device\_count∗4` ；对CPU， :math:`num\_threads=CPU\_NUM∗4` 。在 ``ParallelExecutor`` 中有关于 :math:`CPU\_NUM` 的详细解释。如果没有设置 :math:`CPU\_NUM` ， ``ParallelExecutor`` 可以通过调用 ``multiprocessing.cpu_count()`` 获取CPU数目(cpu count)。默认值为0。












.. _cn_api_fluid_executor:

Executor
-------------------------------


.. py:class:: paddle.fluid.Executor (place)




执行引擎（Executor）使用python脚本驱动，仅支持在单GPU环境下运行。多卡环境下请参考 ``ParallelExecutor`` 。
Python Executor可以接收传入的program,并根据feed map(输入映射表)和fetch_list(结果获取表)
向program中添加feed operators(数据输入算子)和fetch operators（结果获取算子)。
feed map为该program提供输入数据。fetch_list提供program训练结束后用户预期的变量（或识别类场景中的命名）。

应注意，执行器会执行program中的所有算子而不仅仅是依赖于fetch_list的那部分。

Executor将全局变量存储到全局作用域中，并为临时变量创建局部作用域。
当每一mini-batch上的前向/反向运算完成后，局部作用域的内容将被废弃，
但全局作用域中的变量将在Executor的不同执行过程中一直存在。

program中所有的算子会按顺序执行。

参数:	
    - **place** (core.CPUPlace|core.CUDAPlace(n)) – 指明了 ``Executor`` 的执行场所



提示：你可以用 ``Executor`` 来调试基于并行GPU实现的复杂网络，他们有完全一样的参数也会产生相同的结果。


.. py:method:: close()


关闭这个执行器(Executor)。调用这个方法后不可以再使用这个执行器。 对于分布式训练, 该函数会释放在PServers上涉及到目前训练器的资源。
   
**示例代码**

..  code-block:: python
    
    cpu = core.CPUPlace()
    exe = Executor(cpu)
    ...
    exe.close()


.. py:method:: run(program=None, feed=None, fetch_list=None, feed_var_name='feed', fetch_var_name='fetch', scope=None, return_numpy=True,use_program_cache=False)


调用该执行器对象的此方法可以执行program。通过feed map提供待学习数据，以及借助fetch_list得到相应的结果。
Python执行器(Executor)可以接收传入的program,并根据输入映射表(feed map)和结果获取表(fetch_list)
向program中添加数据输入算子(feed operators)和结果获取算子（fetch operators)。
feed map为该program提供输入数据。fetch_list提供program训练结束后用户预期的变量（或识别类场景中的命名）。

应注意，执行器会执行program中的所有算子而不仅仅是依赖于fetch_list的那部分。

参数：  
	- **program** (Program) – 需要执行的program,如果没有给定那么默认使用default_main_program
	- **feed** (dict) – 前向输入的变量，数据,词典dict类型, 例如 {“image”: ImageData, “label”: LableData}
	- **fetch_list** (list) – 用户想得到的变量或者命名的列表, run会根据这个列表给与结果
	- **feed_var_name** (str) – 前向算子(feed operator)变量的名称
	- **fetch_var_name** (str) – 结果获取算子(fetch operator)的输出变量名称
	- **scope** (Scope) – 执行这个program的域，用户可以指定不同的域。缺省为全局域
	- **return_numpy** (bool) – 如果为True,则将结果张量（fetched tensor）转化为numpy
	- **use_program_cache** (bool) – 当program较上次比没有改动则将其置为True
	
返回:	根据fetch_list来获取结果

返回类型:	list(numpy.array)


**示例代码**

..  code-block:: python


	data = layers.data(name='X', shape=[1], dtype='float32')
	hidden = layers.fc(input=data, size=10)
	layers.assign(hidden, out)
	loss = layers.mean(out)
	adam = fluid.optimizer.Adam()
	adam.minimize(loss)


..  code-block:: python
	
	
	cpu = core.CPUPlace()
	exe = Executor(cpu)
	exe.run(default_startup_program())
	
..  code-block:: python
	
	x = numpy.random.random(size=(10, 1)).astype('float32')
	outs = exe.run(
		feed={'X': x},
		fetch_list=[loss.name])
	












.. _cn_api_fluid_global_scope:

global_scope
-------------------------------

.. py:function:: paddle.fluid.global_scope()


获取全局/默认作用域实例。很多api使用默认 ``global_scope`` ，例如 ``Executor.run`` 。

返回：全局/默认作用域实例

返回类型：Scope







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
	x.data = [[1, 2], [3, 4], // seq 1

		  [5, 6], [7, 8], [9, 10]] // seq 2

	x.shape = [5, 2]


LoD可以有多个level(例如，一个段落可以有多个句子，一个句子可以有多个单词)。下面的例子中，Y为LoDTensor ，lod_level为2。表示有2个序列，第一个序列的长度是2(有2个子序列)，第二个序列的长度是1。第一序列的两个子序列长度分别为2和2。第二个序列的子序列的长度是3。


::

	y.lod = [[2 1], [2 2 3]] y.shape = [2+2+3, ...]


.. note::

	在上面的描述中，LoD是基于长度的。在paddle内部实现中，lod是基于偏移的。因此,在内部,y.lod表示为[[0,2,3]，[0,2,4,7]](基于长度的Lod表示为为[[2-0,3-2]，[2-0,4-2,7-4]])。

	可以将LoD理解为recursive_sequence_length（递归序列长度）。此时，LoD必须是基于长度的。由于历史原因。当LoD在API中被称为lod时，它可能是基于偏移的。用户应该注意。




.. py:method::	has_valid_recursive_sequence_lengths(self: paddle.fluid.core.LoDTensor) → bool

.. py:method::	lod(self: paddle.fluid.core.LoDTensor) → List[List[int]]

.. py:method::	recursive_sequence_lengths(self: paddle.fluid.core.LoDTensor) → List[List[int]]

.. py:method::	set_lod(self: paddle.fluid.core.LoDTensor, arg0: List[List[int]]) → None

.. py:method::	set_recursive_sequence_lengths(self: paddle.fluid.core.LoDTensor, arg0: List[List[int]]) → None











.. _cn_api_fluid_LoDTensorArray:

LoDTensorArray
-------------------------------

.. py:class:: paddle.fluid.LoDTensorArray

.. py:method:: append(self: paddle.fluid.core.LoDTensorArray, arg0: paddle.fluid.core.LoDTensor) → None









.. _cn_api_fluid_memory_optimize:

memory_optimize
-------------------------------

.. py:function:: paddle.fluid.memory_optimize(input_program, skip_opt_set=None, print_log=False, level=0, skip_grads=False)


通过重用var内存来优化内存。

.. note::
    它不支持block中嵌套子block。

参数:
	- **input_program** (str) – 输入Program。
	- **skip_opt_set** (set) – set中的vars将不被内存优化。
	- **print_log** (bool) – 是否打印debug日志。
	- **level** (int)  如果 level=0 并且shape是完全相等，则重用。
	
返回: None








.. _cn_api_fluid_name_scope:

name_scope
-------------------------------

.. py:function:: paddle.fluid.name_scope(*args, **kwds)


为operators生成层次名称前缀

注意： 这个函数只能用于调试和可视化。不要将其用于分析，比如graph/program转换。

参数： 
	- **prefix** (str) - 前缀

**示例代码**

.. code-block:: python
          
    with name_scope("encoder"):
        ...
    with name_scope("decoder"):
        ...
    with name_scope("attention"):
        ...







.. _cn_api_fluid_ParallelExecutor:

ParallelExecutor
-------------------------------

.. py:class:: paddle.fluid.ParallelExecutor(use_cuda, loss_name=None, main_program=None, share_vars_from=None, exec_strategy=None, build_strategy=None, num_trainers=1, trainer_id=0, scope=None)




``ParallelExecutor`` 专门设计用来实现数据并行计算，着力于向不同结点(node)分配数据，并行地在不同结点中对数据进行操作。如果在GPU上使用该类运行程序，node则用来指代GPU， ``ParallelExecutor`` 也将自动获取在当前机器上可用的GPU资源。如果在CPU上进行操作，node则指代CPU，同时你也可以通过添加环境变量 ``CPU_NUM`` 来设置CPU设备的个数。例如，``CPU_NUM=4``。但是如果没有设置该环境变量，该类会调用 ``multiprocessing.cpu_count`` 来获取当前系统中CPU的个数。




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

返回类型:	ParallelExecutor

抛出异常：``TypeError`` - 如果提供的参数 ``share_vars_from`` 不是 ``ParallelExecutor`` 类型的，将会弹出此异常

**代码示例**

..  code-block:: python

  train_exe = fluid.ParallelExecutor(use_cuda=True, loss_name=loss.name)
  test_exe = fluid.ParallelExecutor(use_cuda=True,
                                    main_program=test_program,
                                    share_vars_from=train_exe)

  train_loss, = train_exe.run([loss.name], feed=feed_dict)
  test_loss, = test_exe.run([loss.name], feed=feed_dict)



.. py:method::  run(fetch_list, feed=None, feed_dict=None, return_numpy=True)

使用 ``fetch_list`` 执行一个 ``ParallelExecutor`` 对象。

参数 ``feed`` 可以是 ``dict`` 或者 ``list`` 类型变量。如果该参数是 ``dict`` 类型，feed中的数据将会被分割(split)并分送给多个设备（CPU/GPU）。
反之，如果它是 ``list`` ，则列表中的各个元素都会直接分别被拷贝到各设备中。

例如，如果 ``feed`` 是个 ``dict`` 类型变量，则有

..  code-block:: python
    
    exe = ParallelExecutor()
    # 图像会被split到设备中。假设有两个设备，那么每个设备将会处理形为 (24, 1, 28, 28)的图像
    exe.run(feed={'image': numpy.random.random(size=(48, 1, 28, 28))})
  
如果 ``feed`` 是个 ``list`` 类型变量，则有

..  code-block:: python

    exe = ParallelExecutor()
    # 各设备挨个处理列表中的每个元素
    # 第一个设备处理形为 (48, 1, 28, 28) 的图像
    # 第二个设备处理形为 (32, 1, 28, 28) 的图像
    #
    # 使用 exe.device_count 得到设备数目
    exe.run(feed=[{"image": numpy.random.random(size=(48, 1, 28, 28))},
                  {"image": numpy.random.random(size=(32, 1, 28, 28))},
                  ])

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
     1.如果feed参数为dict类型，那么传入 ``ParallelExecutor`` 的数据量 *必须* 大于可用的执行场所数目。否则，C++端将会抛出异常。应额外注意核对数据集的最后一个batch是否比可用执行场所数目大。
     2.如果可用执行场所大于一个，则为每个变量最后获取的结果都是list类型，且这个list中的每个元素都是各个可用执行场所的变量

**代码示例**

..  code-block:: python

        pe = fluid.ParallelExecutor(use_cuda=use_cuda,
                                    loss_name=avg_cost.name,
                                    main_program=fluid.default_main_program())
        loss = pe.run(feed=feeder.feed(cur_batch),
                      fetch_list=[avg_cost.name]))









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

..  code-block:: python

   w_param_attrs = fluid.ParamAttr(name="fc_weight",
                                   learning_rate=0.5,
                                   regularizer=fluid.L2Decay(1.0),
                                   trainable=True)
   y_predict = fluid.layers.fc(input=x, size=10, param_attr=w_param_attrs)













.. _cn_api_fluid_Program:

Program
-------------------------------

.. py:function::  paddle.fluid.Program


创建python program， 在paddleFluid内部会被转换为ProgramDesc描述语言，是被用来创建c++ Program。Program像容器一样也是一种独立的程序语言。Program包括至少一个块（Block），控制流比如conditional_block包括while_op，该Program将会含有嵌套块（nested block）。详情请参阅framework.proto。

注意：默认情况下，paddleFluid内部默认含有 ``default_startup_program`` 和 ``default_main_program`` ，它们将共享参数。 ``default_startup_program`` 只运行一次来初始化参数， ``default_main_program`` 在每个mini batch中运行并调整权重。

返回： empty program

**代码示例**

..  code-block:: python

  main_program = fluid.Program()
  startup_program = fluid.Program()
  with fluid.program_guard(main_program=main_program, startup_program=startup_program):
        fluid.layers.data(name="x", shape=[-1, 784], dtype='float32')
        fluid.layers.data(name="y", shape=[-1, 1], dtype='int32')
        fluid.layers.fc(name="fc", shape=[10], dtype='float32', act="relu")



.. py:attribute:: op_role

operator的角色，值只能是枚举变量{Forward, Backward, Optimize}。

注意：这是一个底层API。它仅用于 ``ParallelExecutor`` 复制或调度operator到设备。

例如，Forward operator应该在每个设备上执行。Backward operator在每个设备上执行，并将后向传播的参数梯度(使用 ``op_role_var`` 获得该变量)合并到一个设备上。Optimize operator只在一个设备上执行，并向其他设备广播新的参数，



.. py:attribute:: set_op_role

operator的角色，值只能是枚举变量{Forward, Backward, Optimize}。

注意：这是一个底层API。它仅用于 ``ParallelExecutor`` 复制或调度operator到设备上执行。

例如，Forward operator应该在每个设备上执行。Backward operato应该在每个设备上执行，并将后向传播的参数梯度(使用op_role_var获得该变量)合并到一个设备上。Optimize operator只在一个设备上执行，并向其他设备广播新的参数



.. py:attribute:: op_role_var

``op_role`` 的辅助变量。

参考: ``Program.op_role`` 文档。

注意:这是一个底层API，用户不应该直接使用它。



.. py:attribute:: set_op_role_var

``op_role`` 的辅助变量。

参考: ``Program.op_role`` 文档。

注意:这是一个底层API。用户不应该直接使用它。



.. py:method:: to_string(throw_on_error, with_details=False)

用于debug

参数：  
	- **throw_on_error** (bool): 没有设置任何必需的字段时，抛出值错误。
	- **with_details** (bool): 值为true时，打印更多关于变量和参数的信息，如trainable, optimize_attr等

返回：(str): debug 字符串

抛出异常： ``ValueError`` - 当 ``throw_on_error == true`` ，但没有设置任何必需的字段时，抛出 ``ValueError`` 。



.. py:method:: clone(for_test=False)

创建一个新的、相同的Program。

有些operator，在训练和测试之间的行为是不同的，比如batch_norm。它们有一个属性is_test来控制行为。当for_test=True时，此方法将把它们的is_test属性更改为True。

- 克隆Program，该Program用于训练时，将 ``for_test`` 设置为False。
- 克隆Program，该Program用于测试时，将 ``for_test`` 设置为True。

注意:此API不会删除任何操作符。请在backward和optimization之前使用clone(for_test=True)。

**代码示例**

..  code-block:: python

  test_program = fluid.default_main_program().clone(for_test=True)
  optimizer = fluid.optimizer.Momentum(learning_rate=0.01, momentum=0.9)
  optimizer.minimize()

参数：
	- **for_test** (bool) – 取值为True时，clone方法内部会把operator的属性 ``is_test`` 设置为 True

返回：一个新的、相同的Program

返回类型:Program

**代码示例**

1.克隆一个Program，示例代码如下：

..  code-block:: python

  train_program = fluid.Program()
  startup_program = fluid.Program()
  with fluid.program_guard(train_program, startup_program):
        img = fluid.layers.data(name='image', shape=[784])
        hidden = fluid.layers.fc(input=img, size=200, act='relu')
        hidden = fluid.layers.dropout(hidden, dropout_prob=0.5)
        loss = fluid.layers.cross_entropy(
                     input=fluid.layers.fc(hidden, size=10, act='softmax'),
                     label=fluid.layers.data(name='label', shape=[1], dtype='int64'))
  test_program = train_program.clone(for_test=True)
  sgd = fluid.optimizer.SGD(learning_rate=1e-3)
  with fluid.program_guard(train_program, startup_program):
        sgd.minimize(loss)    
	
2.如果分别运行 train Program 和 test Program，则可以不使用clone。

..  code-block:: python

	import paddle.fluid as fluid

 	def network(is_test):
	     img = fluid.layers.data(name='image', shape=[784])
	     hidden = fluid.layers.fc(input=img, size=200, act='relu')
	     hidden = fluid.layers.dropout(hidden, dropout_prob=0.5, is_test=is_test)
	     loss = fluid.layers.cross_entropy(
			 input=fluid.layers.fc(hidden, size=10, act='softmax'),
			 label=fluid.layers.data(name='label', shape=[1], dtype='int64'))
	     return loss

	 train_program = fluid.Program()
	 startup_program = fluid.Program()
	 test_program = fluid.Program()

	 with fluid.program_guard(train_program, startup_program):
	     with fluid.unique_name.guard():
		 loss = network(is_test=False)
		 sgd = fluid.optimizer.SGD(learning_rate=1e-3)
		 sgd.minimize(loss)

	 # 不使用测试阶段的startup program
	 with fluid.program_guard(test_program, fluid.Program()):
	     with fluid.unique_name.guard():
		 loss = network(is_test=True)

上边两个代码片段生成的Program是一样的。

.. py:staticmethod:: parse_from_string(binary_str)

反序列化protobuf，转换成program

注意:在序列化和反序列化之后，所有关于参数的信息都会丢失。

参数:	
    - **binary_str_type** (str) – prootbuf二进制字符串

返回:	反序列化后的ProgramDesc

返回类型：Program

.. py:attribute:: num_blocks

该program中的block的个数

.. py:attribute:: random_seed


程序中随机运算符的默认随机种子。0意味着从随机设备中获取随机种子。

注意：必须在operator被添加之前设置。

.. py:method:: global_block()

获取该program的第一个block。

.. py:method:: block(index)

返回该program中 ， ``index`` 指定的block。 ``index`` 类型为int

返回：index对应的block

返回类型：Block

.. py:method:: current_block()

获取当前block。当前block是用来添加operators。

.. py:method:: list_vars()

获取当前program中所有变量。返回值是一个可迭代对象（iterable object)。

返回：generator 会yield每个Program中的变量

返回类型：iterable
	







.. _cn_api_fluid_program_guard:

program_guard
-------------------------------

.. py:function:: paddle.fluid.program_guard(*args, **kwds)



该函数应配合使用python的“with”语句来改变全局主程序(main program)和启动程序(startup program)。

“with”语句块中的layer函数将在新的main program（主程序）中添加operators（算子）和variables（变量）。

**代码示例**

..  code-block:: python

	import paddle.fluid as fluid
	main_program = fluid.Program()
	startup_program = fluid.Program()
	with fluid.program_guard(main_program, startup_program):
		data = fluid.layers.data(...)
 		hidden = fluid.layers.fc(...)

需要注意的是，如果用户不需要构建自己的启动程序或者主程序，一个临时的program将会发挥作用。

**代码示例**

..  code-block:: python

	import paddle.fluid as fluid
	main_program = fluid.Program()
	# 如果您不需要关心startup program,传入一个临时值即可
	with fluid.program_guard(main_program, fluid.Program()):
		data = ...


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














.. _cn_api_fluid_Scope:

Scope
-------------------------------

.. py:class:: paddle.fluid.scope(scope)

(作用域)Scope为变量名的联合。所有变量都属于Scope。

从本地作用域中可以拉取到其双亲作用域的变量。

要想运行一个网络，需要指明它运行所在的域，确切的说： exe.Run(&scope) 。

一个网络可以在不同域上运行，并且更新该域的各类变量。

在作用域上创建一个变量，并在域中获取。

**代码示例**

..  code-block:: python

    # create tensor from a scope and set value to it.
    param = scope.var('Param').get_tensor()
    param_array = np.full((height, row_numel), 5.0).astype("float32")
    param.set(param_array, place)


.. py:method:: drop_kids(self: paddle.fluid.core.Scope) → None
.. py:method:: find_var(self: paddle.fluid.core.Scope, arg0: unicode) → paddle.fluid.core.Variable
.. py:method:: new_scope(self: paddle.fluid.core.Scope) → paddle.fluid.core.Scope
.. py:method:: var(self: paddle.fluid.core.Scope, arg0: unicode) → paddle.fluid.core.Variable   








.. _cn_api_fluid_scope_guard:

scope_guard
-------------------------------

.. py:function:: paddle.fluid.scope_guard(*args, **kwds)


修改全局/默认作用域（scope）,  运行时中的所有变量都将分配给新的scope。

参数：
	- **scope** - 新的全局/默认 scope。

**代码示例**

..  code-block:: python

	import paddle.fluid as fluid
	
	new_scope = fluid.Scope()
	with fluid.scope_guard(new_scope):
		...








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

..  code-block:: python

	data = fluid.layers.data(name="data", shape=[3, 32, 32], dtype="float32")
	fc = fluid.layers.fc(input=data,
			     size=1000,
			     param_attr=WeightNormParamAttr(
				  dim=None,
				  name='weight_norm_param'))








