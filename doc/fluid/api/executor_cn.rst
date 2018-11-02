.. cn_api_fluid_executor:

Executor
=======================


*class* paddle.fluid.executor. Executor *(place)*
---------------------------------------------------------



该 ``Executor`` 是Python实现的类，仅支持在单个GPU环境中运算。对于在多卡环境下的运算，请参照 ``ParallelExecutor`` 。
Python执行器(Executor)可以接收传入的program,并根据输入映射表(feed map)和结果获取表(fetch_list)
向program中添加数据输入算子(feed operators)和结果获取算子（fetch operators)。
feed map为该program提供输入数据。fetch_list提供program训练结束后用户预期的变量（或识别类场景中的命名）。

应注意，执行器会执行program中的所有算子而不仅仅是依赖于fetch_list的那部分。

执行器将全局变量存储到全局域中，并为临时变量创建局部域。
每一小批(minibatch)上进行的前向/后向算法执行完毕后局部域的内容将被作废。
但是全局域中的变量将在执行器不同的执行过程中一直存在。program中所有的算子会按顺序执行。

参数:	
    - place (core.CPUPlace|core.CUDAPlace(n)) – 指明了 ``Executor`` 的执行场所



提示：你可以用Executor来调试基于并行GPU实现的复杂网络，他们有完全一样的参数也会产生相同的结果。


``close()``
++++++++++++++++++++++++

关闭这个执行器(Executor)。调用这个方法后不可以再使用这个执行器。 对于分布式训练, 该函数会释放在PServers上涉及到目前训练器的资源。
   
**示例代码**

..  code-block:: python
    
    cpu = core.CPUPlace()
    exe = Executor(cpu)
    ...
    exe.close()



``run(program=None, feed=None, fetch_list=None, feed_var_name='feed', fetch_var_name='fetch', scope=None, return_numpy=True, use_program_cache=False)``
*************************************************************************************************************************************************************************

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
	
