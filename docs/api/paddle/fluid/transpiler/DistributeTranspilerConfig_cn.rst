.. _cn_api_fluid_transpiler_DistributeTranspilerConfig:

DistributeTranspilerConfig
-------------------------------


.. py:class:: paddle.fluid.transpiler.DistributeTranspilerConfig




单机任务切换为分布式任务的配置类，用户可根据需求进行配置，如指定同步/异步训练，指定节点个数及模型切分逻辑。

返回
::::::::::::
None

属性
::::::::::::
slice_var_up (bool)
'''''''''

是否为 Pserver 将 Tensor 切片，默认为 True, bool 类型属性，默认为 True。该参数将指定是否将参数/梯度切分后均匀分布于多个 PServer 上。slice_var_up 为 True 的情况下，会将参数均匀切分后分布于多个 PServer 端，使每个 PServer 的负载相对均衡。


split_method (PSDispatcher)
'''''''''

参数分发的方式，当前支持的方法包括 :ref:`cn_api_fluid_transpiler_RoundRobin` 和 :ref:`cn_api_fluid_transpiler_HashName` 两种，默认为 RoundRobin。

注意：尝试选择最佳方法来达到负载均衡。

min_block_size (int)
'''''''''

参数切片时，最小数据块的大小，默认为 8192。

注意：根据：https://github.com/PaddlePaddle/Paddle/issues/8638#issuecomment-369912156，当数据块大小超过 2MB 时，我们可以有效地使用带宽。如果你想更改它，请详细查看 slice_variable 函数。

**代码示例**

.. code-block:: python

        from paddle.fluid.transpiler.ps_dispatcher import RoundRobin
        import paddle.fluid as fluid

        config = fluid.DistributeTranspilerConfig()
        config.slice_var_up = True
        config.split_method = RoundRobin
        config.min_block_size = 81920
