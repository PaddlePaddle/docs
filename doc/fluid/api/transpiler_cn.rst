

.. _cn_api_fluid_transpiler_DistributeTranspilerConfig:

RoundRobin

>>>>>>>>>>>>

.. py:class:: paddle.fluid.transpiler.DistributeTranspilerConfig

使用 ``RondRobin`` 方法将变量分配给服务器端点。

RondRobin  `https://en.wikipedia.org/wiki/Round-robin_scheduling <https://en.wikipedia.org/wiki/Round-robin_scheduling>`_  

参数:
  - **pserver_endpoints** (list) - endpoint （ip:port）的 list 
 
.. _cn_api_fluid_DistributeTranspilerConfig:

DistributeTranspilerConfig
>>>>>>>>>>>>

.. py:class:: paddle.fluid.transpiler.DistributeTranspilerConfig

slice_var_up (bool): 使用Tensor切片保存, 默认为True

split_method (PSDispatcher): 可使用 RoundRobin 或者 HashName

  注意: 尝试选择最佳方法来达到负载均衡。


.. py:method:: min_block_size (int): 最小数据块的大小

注意: 根据：https：//github.com/PaddlePaddle/Paddle/issues/8638#issuecomment-369912156, 当数据块大小超过2MB时，我们可以有效地使用带宽。如果你想更改它，请详细查看slice_variable函数。

 
