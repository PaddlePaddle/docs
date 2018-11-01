.. cn_api_fluid_executor

Executor
=======================


*class* paddle.fluid.executor. Executor *(place)*
---------------------------------------------------------

.. An Executor in Python, only support the single-GPU running. For multi-cards, please refer to ParallelExecutor.
.. Python executor takes a program, add feed operators and fetch operators to this program according to feed map and fetch_list. 
.. Feed map provides input data for the program. fetch_list provides the variables(or names) that user want to get after program run.
.. Note: the executor will run all operators in the program but not only the operators dependent by the fetch_list.
.. It store the global variables into the global scope, and create a local scope for the temporary variables. 
.. The local scope contents will be discarded after every minibatch forward/backward finished.
.. But the global scope variables will be persistent through different runs. All of ops in program will be running in sequence.


该 ``Executor`` 是Python实现的类，仅支持在单个GPU环境中运算。对于在多卡环境下的运算，请参照 ``ParallelExecutor`` 。
Python执行器(Executor)可以接收传入的program,并根据输入映射表(feed map)和结果获取表(fetch_list)
向program中添加数据输入算子(feed operators)和结果获取算子（fetch operators)。
feed map为该program提供输入数据。fetch_list提供program训练结束后用户预期的变量（或识别类场景中的命名）。

应注意，执行器会执行program中的所有算子而不仅仅是依赖于fetch_list的那部分。

执行器将全局变量存储到全局域中，并为临时变量创建局部域。
每一小批(minibatch)前向/后向算法执行完毕后局部域的内容将被作废。
但是全局域中的变量将在执行器不同的执行过程中一直存在。program中所有的算子会按顺序执行。

参数:	
    - place (core.CPUPlace|core.CUDAPlace(n)) – 指明了 ``Executor`` 执行场所

.. Note: For debugging complicated network in parallel-GPUs, you can test it on the executor.
.. They has the exactly same arguments, and expected the same results.

提示：你可以用Executor来调试基于并行GPU实现的复杂网络，他们有完全一样的参数也会产生相同的结果。


``close()``
++++++++++++++++++++++++

关闭这个执行器(Executor)。调用这个方法后不可以再使用这个执行器。 对于分布式训练, 该函数会释放在PServers上涉及到目前训练器的资源。
   
**示例代码**

..  code-block:: python
    
    >>> cpu = core.CPUPlace()
    >>> exe = Executor(cpu)
    >>> ...
    >>> exe.close()



``run(program=None, feed=None, fetch_list=None, feed_var_name='feed', fetch_var_name='fetch', scope=None, return_numpy=True, use_program_cache=False)``
*************************************************************************************************************************************************************************

调用该执行器对象的此方法可以执行program。通过feed map提供待学习数据，以及借助fetch_list得到相应的结果。
Python执行器(Executor)可以接收传入的program,并根据输入映射表(feed map)和结果获取表(fetch_list)
向program中添加数据输入算子(feed operators)和结果获取算子（fetch operators)。
feed map为该program提供输入数据。fetch_list提供program训练结束后用户预期的变量（或识别类场景中的命名）。

应注意，执行器会执行program中的所有算子而不仅仅是依赖于fetch_list的那部分。

参数：  
		- program (Program) – 需要执行的program,如果没有给定那么默认使用default_main_program
    - feed (dict) – 输入变量的映射词典, 例如 {“image”: ImageData, “label”: LableData}
    - fetch_list (list) – 用户想得到的变量或者命名的列表, run会根据这个列表给与结果.
    - feed_var_name (str) – the name for the input variable of feed Operator.
    - fetch_var_name (str) – the name for the output variable of fetch Operator.
    - scope (Scope) – the scope used to run this program, you can switch it to different scope. default is global_scope
    - return_numpy (bool) – if convert the fetched tensor to numpy
    - use_program_cache (bool) – set use_program_cache to true if program not changed compare to the last step.
