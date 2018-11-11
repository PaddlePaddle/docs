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
		- main_program (Program) – “with”语句中将使用的新的main program。
		- startup_program (Program) – “with”语句中将使用的新的startup program。若传入 ``None`` 则不改变当前的启动程序。




.. _cn_api_fluid_executor:

Executor
>>>>>>>>>>>>>>>>>>>>>


.. py:class:: class paddle.fluid. Executor (place)




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
    - place (core.CPUPlace|core.CUDAPlace(n)) – 指明了 ``Executor`` 的执行场所



提示：你可以用Executor来调试基于并行GPU实现的复杂网络，他们有完全一样的参数也会产生相同的结果。


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
	- program (Program) – 需要执行的program,如果没有给定那么默认使用default_main_program
	- feed (dict) – 前向输入的变量，数据,词典dict类型, 例如 {“image”: ImageData, “label”: LableData}
	- fetch_list (list) – 用户想得到的变量或者命名的列表, run会根据这个列表给与结果
	- feed_var_name (str) – 前向算子(feed operator)变量的名称
	- fetch_var_name (str) – 结果获取算子(fetch operator)的输出变量名称
	- scope (Scope) – 执行这个program的域，用户可以指定不同的域。缺省为全局域
	- return_numpy (bool) – 如果为True,则将结果张量（fetched tensor）转化为numpy
	- use_program_cache (bool) – 当program较上次比没有改动则将其置为True
	
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

.. py:class:: class paddle.fluid.DistributeTranspiler (config=None)


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
	- trainer_id (int) – 当前Trainer worker的id, 如果有n个Trainer worker, id 取值范围为0 ~ n-1
	- program (Program|None) – 待transpile（转译）的program, 缺省为 ``fluid.default_main_program()`` 
	- pservers (str) – 内容为Pserver列表的字符串，格式为：按逗号区分不同的Pserver，每个Pserver的格式为 *ip地址:端口号* 
	- trainers (int|str) – 在Pserver模式下，该参数指Trainer机的个数；在nccl2模式下，它是一个内容为Trainer终端列表的字符串
	- sync_mode (bool) – 是否做同步训练(synchronous training), 默认为True
 	- startup_program (Program|None) – 待transpile（转译）的startup_program，默认为 ``fluid.default_main_program()``
	- current_endpoint (str) – 当需要把program转译（transpile）至NCCL2模式下时，需要将当前endpoint（终端）传入该参数。Pserver模式不使用该参数

.. py:method:: get_trainer_program(wait_port=True)


该方法可以得到Trainer侧的program。

返回:	Trainer侧的program

返回类型:	Program



.. py:method:: get_pserver_program(endpoint)


该方法可以得到Pserver（参数服务器）侧的程序
 
参数:	
	- endpoint (str) – 当前Pserver终端
 
返回:	当前Pserver需要执行的program

返回类型:	Program


.. py:method:: get_pserver_programs(endpoint)


该方法可以得到Pserver侧用于分布式训练的 ``main_program`` 和 ``startup_program`` 。

参数:	
	- endpoint (str) – 当前Pserver终端

返回:	(main_program, startup_program), “Program”类型的元组

返回类型:	tuple 
 
 
.. py:method:: get_startup_program(endpoint, pserver_program=None, startup_program=None)


**该函数已停止使用**
获取当前Pserver的startup_program，如果有多个被分散到不同blocks的变量，则修改operator的输入变量。

参数:	
	- endpoint (str) – 当前Pserver终端
	- pserver_program (Program) – 已停止使用。 先调用get_pserver_program
 	- startup_program (Program) – 已停止使用。应在初始化时传入startup_program

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
    - input_program (Program) – 在此program中插入 ``delete_op`` 
    - skip_opt_set (set) – 在内存优化时跳过的变量的集合

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
     	 那么数据可以是一个numpy数组，形状为（5,1）。同时， ``recursive_seq_lens`` 为 [[2, 3]]，表明各个句子的长度。这个长度为基准的``recursive_seq_lens`` 将在函数中会被转化为以偏移量为基准的 LoD [[0, 2, 5]]。
     	 请参照 ``api_guide_low_level_lod_tensor`` 来获取更多LoD的详细介绍。

参数:
	- data (numpy.ndarray|list|LoDTensor) – 容纳着待复制数据的一个numpy数组、列表或LoD Tensor
	- recursive_seq_lens (list) – 一组列表的列表， 表明了由用户指明的length-based level of detail信息
	- place (Place) – CPU或GPU。 指明返回的新LoD Tensor存储地点

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
    - recursive_seq_lens (list) – 一组列表的列表， 表明了由用户指明的length-based level of detail信息
    - base_shape (list) – LoDTensor所容纳的基本元素的形状
    - place (Place) –  CPU或GPU。 指明返回的新LoD Tensor存储地点
    - low (int) – 随机数下限
    - high (int) – 随机数上限

返回:	一个fluid LoDTensor对象，包含数据和recursive_seq_lens信息







.. _cn_api_fluid_ParamAttr:

 
ParamAttr
>>>>>>>>>>>>>>>>>>>>>>>>>


.. py:class:: class paddle.fluid.ParamAttr(name=None, initializer=None, learning_rate=1.0, regularizer=None, trainable=True, gradient_clip=None, do_model_average=False)

该类代表了参数的各种属性。 为了使神经网络训练过程更加流畅，用户可以根据需要调整参数属性。比如learning rate（学习率）, regularization（正则化）, trainable（可训练性）, do_model_average(平均化模型)和参数初始化方法.

参数:	
    - name (str) – 参数名。默认为None。
    - initializer (Initializer) – 初始化该参数的方法。 默认为None
    - learning_rate (float) – 参数的学习率。计算方法为 global_lr*parameter_lr∗scheduler_factor。 默认为1.0
    - regularizer (WeightDecayRegularizer) – 正则因子. 默认为None
    - trainable (bool) – 该参数是否可训练。默认为True
    - gradient_clip (BaseGradientClipAttr) – 减少参数梯度的方法。默认为None
    - do_model_average (bool) – 该参数是否服从模型平均值。默认为False
    
**代码示例**

..  code-block:: python

   w_param_attrs = fluid.ParamAttr(name="fc_weight",
                                   learning_rate=0.5,
                                   regularizer=fluid.L2Decay(1.0),
                                   trainable=True)
   y_predict = fluid.layers.fc(input=x, size=10, param_attr=w_param_attrs)










