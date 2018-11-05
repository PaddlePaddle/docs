.. _cn_api_fluid_default_startup_program:




default_startup_program
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

paddle.fluid.default_startup_program()
""""""""""""""""""""""""""""""""""""""""""


该函数可以获取默认/全局 startup program (启动程序)。

``fluid.layers`` 中的layer函数会新建参数、readers(读取器)、NCCL句柄作为全局变量。 

startup_program会使用内在的operators（算子）去初始化他们，并由layer函数将这些operators追加到startup program中。

该函数将返回默认的或当前的startup_program。用户可以使用 ``fluid.program_guard`` 去切换program。

返回:	startup program

返回类型:	Program





.. _cn_api_fluid_default_main_program:

default_main_program
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

paddle.fluid.default_main_program()
""""""""""""""""""""""""""""""""""""""""""




此函数用于获取默认或全局main program(主程序)。该主程序用于训练和测试模型。

``fluid.layers`` 中的所有layer函数可以向 ``default_main_program`` 中添加operators（算子）和variables（变量）。

``default_main_program`` 是fluid的许多编程接口（API）的Program参数的缺省值。例如,当用户program没有传入的时候，
``Executor.run()`` 会默认执行 ``default_main_program`` 。


返回：	main program

返回类型:	Program





.. _cn_api_fluid_program_guard:

program_guard
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

paddle.fluid.program_guard(*args, **kwds)
""""""""""""""""""""""""""""""""""""""""""


该函数应配合使用python的“with”语句来改变全局主程序(main program)和启动程序(startup program)。

“with”语句块中的layer函数将在新的main program（主程序）后添加operators（算子）和variables（变量）。

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






.. _cn_api_fluid_DistributeTranspiler:

DistributeTranspiler
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

 *class* paddle.fluid.DistributeTranspiler *(config=None)*
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

该类可以把fluid program转变为分布式数据并行计算程序（distributed data-parallelism programs）,可以有Pserver和NCCL2两种模式。
当program在Pserver（全称：parameter server）模式下， ``main_program`` (主程序)转为使用一架远程parameter server(即pserver,参数服务器)来进行参数优化，并且优化图会被输入到一个pserver program中。
在NCCL2模式下，transpiler会在 ``startup_program`` 中附加一个 ``NCCL_ID`` 广播算子（broadcasting operators）来实现在该集群中所有工作结点共享
 ``NCCL_ID`` 。
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



transpile(trainer_id, program=None, pservers='127.0.0.1:6174', trainers=1, sync_mode=True, startup_program=None, current_endpoint='127.0.0.1:6174')
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
该方法可以运行该transpiler（转译器）。

参数:	

    - trainer_id (int) – 当前Trainer worker的id, 如果有n个Trainer worker, id 取值范围为0 ~ n-1
    - program (Program|None) – 待transpile（转译）的program, 缺省为 ``fluid.default_main_program()`` 
    - pservers (str) – 内容为Pserver列表的字符串，格式为：按逗号区分不同的Pserver，每个Pserver的格式为 *ip地址:端口号* 
    - trainers (int|str) – 在Pserver模式下，该参数指Trainer机的个数；在nccl2模式下，它是一个内容为Trainer终端列表的字符串
    - sync_mode (bool) – 是否做同步训练(synchronous training), 默认为True
    - startup_program (Program|None) – startup_program to transpile, default is fluid.default_main_program()
    - current_endpoint (str) – 当需要把program转译（transpile）至NCCL2模式下时，需要将当前endpoint（终端）传入该参数。Pserver模式不使用该参数

get_trainer_program(wait_port=True)
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
该方法可以得到Trainer侧的program。
返回:	Trainer侧的program
返回类型:	Program


get_pserver_program(endpoint)
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
该方法可以得到Pserver（参数服务器）侧的程序
 
参数:	
 	- endpoint (str) – 当前Pserver终端
 
返回:	当前Pserver需要执行的program
返回类型:	Program


get_pserver_programs(endpoint)
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
该方法可以得到Pserver侧用于分布式训练的 ``main_program`` 和 ``startup_program`` 。
参数:	
	- endpoint (str) – 当前Pserver终端
返回:	(main_program, startup_program), “Program”类型的元组
返回类型:	tuple 
 
get_startup_program(endpoint, pserver_program=None, startup_program=None)
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
**该函数已停止使用**
返回当前Pserver的startup_program。如果由多个被分散到不同blocks的变量，则修改operator。
参数:	

	endpoint (str) – 当前Pserver终端
	pserver_program (Program) – 已停止使用, 先调用get_pserver_program
 	startup_program (Program) – 已停止使用, 应在初始化时传入startup_program

返回:	Pserver侧的startup_program
返回类型:	Program



.. _cn_api_fluid_release_memory:

release_memory
>>>>>>>>>>>>>>>>>>>>>>>>>>>

paddle.fluid.release_memory(input_program, skip_opt_set=None) 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

该函数可以调整输入program，插入 ``delete_op`` 删除算子，提前删除不需要的变量。
改动是在变量本身上进行的。
提醒: 该API还在试验阶段，会在后期版本中删除。不建议用户使用。






.. _cn_api_fluid_ParallelExecutor:

.. _cn_api_fluid_ExecutionStrategy:

.. _cn_api_fluid_BuildStrategy:




.. _cn_api_fluid_create_lod_tensor:


create_lod_tensor
>>>>>>>>>>>>>>>>>>>>>>>>>

paddle.fluid.create_lod_tensor(data, recursive_seq_lens, place) 
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

该函数从一个numpy数组，列表或者已经存在的lod tensor中创建一个lod tensor。
通过一下几步实现:
	1.检查length-based level of detail (LoD,长度为基准的细节层次)，或称recursive_sequence_lengths(递归序列长度)的正确性
	2.将recursive_sequence_lengths转化为offset-based LoD(偏移量为基准的LoD)
        3.把提供的numpy数组，列表或者已经存在的lod tensor复制到CPU或GPU中(看实在什么设备上进行的计算)
        4.利用offset-based LoD来设置LoD
::
     例如：
Suppose we want LoDTensor to hold data for sequences of word, where each word is represented by an integer. If we want to create a LoDTensor to represent two sentences, one of 2 words, and one of 3 words.
Then data can be a numpy array of integers with shape (5, 1). recursive_seq_lens will be [[2, 3]], indicating the length(# of words) in each sentence. This length-based recursive_seq_lens [[2, 3]] will be converted to offset-based LoD [[0, 2, 5]] inside the function call.
Please reference api_guide_low_level_lod_tensor for more details regarding LoD.


参数:
data (numpy.ndarray|list|LoDTensor) – a numpy array or a LoDTensor or a list holding the data to be copied.
recursive_seq_lens (list) – a list of lists indicating the length-based level of detail info specified by the user.
place (Place) – CPU or GPU place indicating where the data in the new LoDTensor will be stored.
返回:
A fluid LoDTensor object with tensor data and recursive_seq_lens info.
















