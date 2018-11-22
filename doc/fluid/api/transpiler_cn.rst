

.. _cn_api_fluid_transpiler_memory_optimize:

memory_optimize
>>>>>>>>>>>>

.. py:class:: paddle.fluid.transpiler.memory_optimize(input_program, skip_opt_set=None, print_log=False, level=0, skip_grads=False)

通过重用var内存来优化内存。

注意:它不支持block中嵌套子block。

参数:

  - **input_program** (str) – 输入Program。
  - **skip_opt_set** (set) – set中的vars将不被内存优化。
  - **print_log** (bool) – 是否打印debug日志。
  - **level** (int) - 如果 level=0 并且shape是完全相等，则重用。
	
返回: None

.. _cn_api_fluid_transpiler_RoundRobin:

RoundRobin

>>>>>>>>>>>>

.. py:class:: paddle.fluid.transpiler.RoundRobin(pserver_endpoints)

使用 ``RondRobin`` 方法将变量分配给服务器端点。

RondRobin  `https://en.wikipedia.org/wiki/Round-robin_scheduling <https://en.wikipedia.org/wiki/Round-robin_scheduling>`_  

参数:
  - **pserver_endpoints** (list) - endpoint （ip:port）的 list 
 
 
.. _cn_api_fluid_transpiler_HashName:

HashName
>>>>>>>>>>>>

.. py:class:: paddle.fluid.transpiler.HashName(pserver_endpoints)

使用 python “Hash()”函数将变量名散列到多个端点。

参数:
  - **pserver_endpoints** (list) - endpoint （ip:port）的 list 


.. _cn_api_fluid_transpiler_DistributeTranspilerConfig:

DistributeTranspilerConfig
>>>>>>>>>>>>

.. py:class:: paddle.fluid.transpiler.DistributeTranspilerConfig

slice_var_up (bool): 使用Tensor切片保存, 默认为True

split_method (PSDispatcher): 可使用 RoundRobin 或者 HashName

  注意: 尝试选择最佳方法来达到负载均衡。


.. py:method:: min_block_size (int): 最小数据块的大小

注意: 根据：https：//github.com/PaddlePaddle/Paddle/issues/8638#issuecomment-369912156, 当数据块大小超过2MB时，我们可以有效地使用带宽。如果你想更改它，请详细查看slice_variable函数。
