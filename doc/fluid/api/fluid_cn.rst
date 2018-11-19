.. _cn_api_fluid_default_startup_program:




default_startup_program
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.default_startup_program()



该函数可以获取默认/全局 startup program (启动程序)。

``fluid.layers`` 中的layer函数会新建参数、readers(读取器)、NCCL句柄作为全局变量。 

startup_program会使用内在的operators（算子）去初始化他们，并由layer函数将这些operators追加到startup program中。

该函数将返回默认的或当前的startup_program。用户可以使用 ``fluid.program_guard`` 去切换program。

返回:	startup program

返回类型:	Program





.. _cn_api_fluid_default_main_program:

default_main_program
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.default_main_program()





此函数用于获取默认或全局main program(主程序)。该主程序用于训练和测试模型。

``fluid.layers`` 中的所有layer函数可以向 ``default_main_program`` 中添加operators（算子）和variables（变量）。

``default_main_program`` 是fluid的许多编程接口（API）的Program参数的缺省值。例如,当用户program没有传入的时候，
``Executor.run()`` 会默认执行 ``default_main_program`` 。


返回：	main program

返回类型:	Program





.. _cn_api_fluid_program_guard:

program_guard
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.program_guard(*args, **kwds)



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
	# does not care about startup program. Just pass a temporary value.
	with fluid.program_guard(main_program, fluid.Program()):
		data = ...


参数：  
		- **main_program** (Program) – “with”语句中将使用的新的main program。
		- **startup_program** (Program) – “with”语句中将使用的新的startup program。若传入 ``None`` 则不改变当前的启动程序。




.. _cn_api_fluid_executor:

Executor
>>>>>>>>>>>>>>>>>>>>>


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
	






.. _cn_api_fluid_DistributeTranspiler:

DistributeTranspiler
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.DistributeTranspiler (config=None)


该类可以把fluid program转变为分布式数据并行计算程序（distributed data-parallelism programs）,可以有Pserver和NCCL2两种模式。
当program在Pserver（全称：parameter server）模式下， ``main_program`` (主程序)转为使用一架远程parameter server(即pserver,参数服务器)来进行参数优化，并且优化图会被输入到一个pserver program中。
在NCCL2模式下，transpiler会在 ``startup_program`` 中附加一个 ``NCCL_ID`` 广播算子（broadcasting operators）来实现在该集群中所有工作结点共享``NCCL_ID`` 。
调用 ``transpile_nccl2`` 后， 你 **必须** 将 ``trainer_id`` , ``num_trainers`` 参数提供给 ``ParallelExecutor`` 来启动NCCL2分布式模式。 




**代码示例**

..  code-block:: python

	# for pserver mode
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

	# for nccl2 mode
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



.. _cn_api_fluid_release_memory:

release_memory
>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.release_memory(input_program, skip_opt_set=None) 


该函数可以调整输入program，插入 ``delete_op`` 删除算子，提前删除不需要的变量。
改动是在变量本身上进行的。
提醒: 该API还在试验阶段，会在后期版本中删除。不建议用户使用。

参数:	
    - **input_program** (Program) – 在此program中插入 ``delete_op`` 
    - **skip_opt_set** (set) – 在内存优化时跳过的变量的集合

返回: None








.. _cn_api_fluid_create_lod_tensor:


create_lod_tensor
>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.create_lod_tensor(data, recursive_seq_lens, place) 


该函数从一个numpy数组，列表或者已经存在的lod tensor中创建一个lod tensor。
通过一下几步实现:
	1. 检查length-based level of detail (LoD,长度为基准的细节层次)，或称recursive_sequence_lengths(递归序列长度)的正确性
	2. 将recursive_sequence_lengths转化为offset-based LoD(偏移量为基准的LoD)
        3. 把提供的numpy数组，列表或者已经存在的lod tensor复制到CPU或GPU中(依据执行场所确定)
        4. 利用offset-based LoD来设置LoD
例如：
         假如我们想用LoD Tensor来承载一词序列的数据，其中每个词由一个整数来表示。现在，我们意图创建一个LoD Tensor来代表两个句子，其中一个句子有两个词，另外一个句子有三个。
     	 那么数据可以是一个numpy数组，形状为（5,1）。同时， ``recursive_seq_lens`` 为 [[2, 3]]，表明各个句子的长度。这个长度为基准的 ``recursive_seq_lens`` 将在函数中会被转化为以偏移量为基准的 LoD [[0, 2, 5]]。
     	 请参照 ``api_guide_low_level_lod_tensor`` 来获取更多LoD的详细介绍。

参数:
	- **data** (numpy.ndarray|list|LoDTensor) – 容纳着待复制数据的一个numpy数组、列表或LoD Tensor
	- **recursive_seq_lens** (list) – 一组列表的列表， 表明了由用户指明的length-based level of detail信息
	- **place** (Place) – CPU或GPU。 指明返回的新LoD Tensor存储地点

返回: 一个fluid LoDTensor对象，包含数据和recursive_seq_lens信息





.. _cn_api_fluid_create_random_int_lodtensor:


create_random_int_lodtensor
>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.create_random_int_lodtensor(recursive_seq_lens, base_shape, place, low, high)



该函数创建一个存储多个随机整数的LoD Tensor。

该函数是经常在书中出现的案例，所以我们根据新的API： ``create_lod_tensor`` 更改它然后放在LoD Tensor板块里来简化代码。

该函数实现以下功能：

    1. 根据用户输入的length-based recursive_seq_lens（基于长度的递归序列长）和在 ``basic_shape`` 中的基本元素形状计算LoDTensor的整体形状
    2. 由此形状，建立numpy数组
    3. 使用API： ``create_lod_tensor`` 建立LoDTensor


假如我们想用LoD Tensor来承载一词序列的数据，其中每个词由一个整数来表示。现在，我们意图创建一个LoD Tensor来代表两个句子，其中一个句子有两个词，另外一个句子有三个。那么 ``base_shape`` 为[1], 输入的length-based ‘recursive_seq_lens’ 是 [[2, 3]]。那么LoDTensor的整体形状应为[5, 1]，即为两个句子存储5个词。

参数:	
    - **recursive_seq_lens** (list) – 一组列表的列表， 表明了由用户指明的length-based level of detail信息
    - **base_shape** (list) – LoDTensor所容纳的基本元素的形状
    - **place** (Place) –  CPU或GPU。 指明返回的新LoD Tensor存储地点
    - **low** (int) – 随机数下限
    - **high** (int) – 随机数上限

返回:	一个fluid LoDTensor对象，包含数据和recursive_seq_lens信息







.. _cn_api_fluid_ParamAttr:

 
ParamAttr
>>>>>>>>>>>>>>>>>>>>>>>>>


.. py:class:: paddle.fluid.ParamAttr(name=None, initializer=None, learning_rate=1.0, regularizer=None, trainable=True, gradient_clip=None, do_model_average=False)

该类代表了参数的各种属性。 为了使神经网络训练过程更加流畅，用户可以根据需要调整参数属性。比如learning rate（学习率）, regularization（正则化）, trainable（可训练性）, do_model_average(平均化模型)和参数初始化方法.

参数:	
    - **name** (str) – 参数名。默认为None。
    - **initializer** (Initializer) – 初始化该参数的方法。 默认为None
    - **learning_rate** (float) – 参数的学习率。计算方法为 global_lr*parameter_lr∗scheduler_factor。 默认为1.0
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







.. _cn_api_fluid_DataFeeder:

DataFeeder
>>>>>>>>>>>>>>>>>

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


弹出异常:	  ``ValueError``  – 如果一些变量不在此 Program 中


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

**特别注意：** 设备（CPU或GPU）的数目必须等于minibatch的数目



.. py:method::  decorate_reader(reader, multi_devices, num_places=None, drop_last=True)


  
将reader返回的输入数据batch转换为多个mini-batch，之后每个mini-batch都会被输入进各个设备（CPU或GPU）中。
    
参数：
        - **reader** (fun) – 待输入的数据
        - **multi_devices** (bool) – 执行场所的数目，默认为None
        - **num_places** (int) – 执行场所的数目，默认为None
        - **drop_last** (bool) – 舍弃数目匹配不上的batch或设备

返回：转换结果

返回类型: dict
    
弹出异常： ValueError – 如果 ``drop_last`` 值为False并且reader返回的minibatch数目与设备数目不相等时，产生此异常


        



.. _cn_api_fluid_BuildStrategy:

BuildStrategy
>>>>>>>>>>>>>>>>>>

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



.. py:method:: debug_graphviz_path

str类型。它表明了以graphviz格式向文件中写入SSA图的路径，有利于调试。 默认值为""。



.. py:method:: fuse_elewise_add_act_ops

bool类型。它表明了是否融合（fuse）elementwise_add_op和activation_op。这会使整体执行过程更快一些。默认为False。



.. py:method:: gradient_scale_strategy

str类型。在 ``ParallelExecutor`` 中，存在三种定义 *loss@grad* 的方式，分别为 ``CoeffNumDevice``, ``One`` 与 ``Customized``。默认情况下， ``ParallelExecutor`` 根据设备数目来设置 *loss@grad* 。如果你想自定义 *loss@grad* ，你可以选择 ``Customized`` 方法。默认为 ``CoeffNumDevice`` 。



.. py:method:: reduce_strategy

str类型。在 ``ParallelExecutor`` 中，存在两种减少策略（reduce strategy），即 ``AllReduce`` 和 ``Reduce`` 。如果你需要在所有执行场所上独立地进行参数优化，可以使用 ``AllReduce`` 。反之，如果使用 ``Reduce`` 策略，所有参数的优化将均匀地分配给不同的执行场所，随之将优化后的参数广播给其他执行场所。在一些模型中， ``Reduce`` 策略执行速度更快一些。默认值为 ``AllReduce`` 。






.. _cn_api_fluid_ExecutionStrategy:

ExecutionStrategy
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

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



.. py:method:: allow_op_delay
   
这是一个bool类型成员，表示是否推迟communication operators(交流运算)的执行，这样做会使整体执行过程更快一些。但是在一些模型中，allow_op_delay会导致程序中断。默认为False。
  


.. py:method:: num_iteration_per_drop_scope
  
int型成员。它表明了清空执行时产生的临时变量需要的程序执行重复次数。因为临时变量的形可能在两次重复过程中保持一致，所以它会使整体执行过程更快。默认值为100。

特别注意：
  1.如果在调用 ``run`` 方法时获取结果数据，``ParallelExecutor`` 会在当前程序重复执行尾部清空临时变量
  
  2.在一些NLP模型里，该成员会致使GPU内存不足。此时，你应减少 ``num_iteration_per_drop_scope`` 的值



.. py:method:: num_threads

int型成员。它代表了线程池(thread pool)的大小。这些线程会被用来执行当前 ``ParallelExecutor`` 的program中的operator（算子，运算）。如果 :math: num_threads=1 ，则所有的operator将一个接一个地执行，但在不同的程序重复周期(iterations)中执行顺序可能不同。如果该成员没有被设置，则在 ``ParallelExecutor`` 中，它会依据设备类型(device type)、设备数目(device count)而设置为相应值。对GPU，:math:`num_threads=device_count∗4` ；对CPU，:math:`num_threads=CPU_NUM∗4` 。在 ``ParallelExecutor`` 中有关于 ``CPU_NUM`` 的详细解释。如果没有设置 ``CPU_NUM`` ， ``ParallelExecutor`` 可以通过调用 ``multiprocessing.cpu_count()`` 获取CPU数目(cpu count)。默认值为0。






.. _cn_api_fluid_ParallelExecutor:

ParallelExecutor
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

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

弹出异常：``TypeError`` - 如果提供的参数 ``share_vars_from`` 不是 ``ParallelExecutor`` 类型的，将会弹出此异常

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

弹出异常： 
         ``ValueError`` - 如果feed参数是list类型，但是它的长度不等于可用设备（执行场所）的数目，再或者给定的feed不是dict类型，弹出此异常
         
         ``TypeError`` - 如果feed参数是list类型，但是它里面的元素不是dict类型时，弹出此异常

额外注意：
     1.如果feed参数为dict类型，那么传入 ``ParallelExecutor`` 的数据量 *必须* 大于可用的执行场所数目。否则，C++端将会弹出异常。应额外注意核对数据集的最后一个batch是否比可用执行场所数目大。
    
     2.如果可用执行场所大于一个，则为每个变量最后获取的结果都是list类型，且这个list中的每个元素都是各个可用执行场所的变量

**代码示例**

..  code-block:: python

        pe = fluid.ParallelExecutor(use_cuda=use_cuda,
                                    loss_name=avg_cost.name,
                                    main_program=fluid.default_main_program())
        loss = pe.run(feed=feeder.feed(cur_batch),
                      fetch_list=[avg_cost.name]))


