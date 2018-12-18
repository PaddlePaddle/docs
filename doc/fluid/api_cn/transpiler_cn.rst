#################
 fluid.transpiler
#################



.. _cn_api_fluid_transpiler_DistributeTranspiler:

DistributeTranspiler
-------------------------------

.. py:class:: paddle.fluid.transpiler.DistributeTranspiler (config=None)


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







.. _cn_api_fluid_transpiler_DistributeTranspilerConfig:

DistributeTranspilerConfig
-------------------------------

.. py:class:: paddle.fluid.transpiler.DistributeTranspilerConfig

.. py:method:: slice_var_up (bool)

为Pserver将张量切片, 默认为True

.. py:method:: split_method (PSDispatcher)

可使用 RoundRobin 或者 HashName

注意: 尝试选择最佳方法来达到负载均衡。


.. py:attribute:: min_block_size (int)

最小数据块的大小

注意: 根据：https://github.com/PaddlePaddle/Paddle/issues/8638#issuecomment-369912156 , 当数据块大小超过2MB时，我们可以有效地使用带宽。如果你想更改它，请详细查看slice_variable函数。







.. _cn_api_fluid_transpiler_HashName:

HashName
-------------------------------

.. py:class:: paddle.fluid.transpiler.HashName(pserver_endpoints)

使用 python ``Hash()`` 函数将变量名散列到多个pserver终端。

参数:
  - **pserver_endpoints** (list) - endpoint （ip:port）的 list 








.. _cn_api_fluid_transpiler_memory_optimize:

memory_optimize
-------------------------------

.. py:function:: paddle.fluid.transpiler.memory_optimize(input_program, skip_opt_set=None, print_log=False, level=0, skip_grads=False)

通过重用var内存来优化内存。

注意:它不支持block中嵌套子block。

参数:
  - **input_program** (str) – 输入Program。
  - **skip_opt_set** (set) – set中的vars将不被内存优化。
  - **print_log** (bool) – 是否打印debug日志。
  - **level** (int) - 如果 level=0 并且shape是完全相等，则重用。

返回: None








.. _cn_api_fluid_transpiler_release_memory:

release_memory
-------------------------------

.. py:function:: paddle.fluid.transpiler.release_memory(input_program, skip_opt_set=None) 


该函数可以调整输入program，插入 ``delete_op`` 删除算子，提前删除不需要的变量。
改动是在变量本身上进行的。

**提醒** : 该API还在试验阶段，会在后期版本中删除。不建议用户使用。

参数:
    - **input_program** (Program) – 在此program中插入 ``delete_op`` 
    - **skip_opt_set** (set) – 在内存优化时跳过的变量的集合

返回: None















.. _cn_api_fluid_transpiler_RoundRobin:

RoundRobin
-------------------------------

.. py:class:: paddle.fluid.transpiler.RoundRobin(pserver_endpoints)

使用 ``RondRobin`` 方法将变量分配给服务器端点。

`RondRobin <https://en.wikipedia.org/wiki/Round-robin_scheduling>`_

参数:
  - **pserver_endpoints** (list) - endpoint （ip:port）的 list 
 
 






