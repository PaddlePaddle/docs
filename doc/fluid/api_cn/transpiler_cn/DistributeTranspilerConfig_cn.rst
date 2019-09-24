.. _cn_api_fluid_transpiler_DistributeTranspilerConfig:

DistributeTranspilerConfig
-------------------------------

.. py:class:: paddle.fluid.transpiler.DistributeTranspilerConfig

单机任务切换为分布式任务的配置类，其中较为重要的几个配置参数如下所示：

.. py:method:: slice_var_up (bool)

是否为Pserver将张量切片, 默认为True

.. py:method:: split_method (PSDispatcher)

参数分发的方式，当前支持的方法包括 :ref:`cn_api_fluid_transpiler_RoundRobin` 和 :ref:`cn_api_fluid_transpiler_HashName` 两种。

注意: 尝试选择最佳方法来达到负载均衡。

.. py:attribute:: min_block_size (int)

参数切片时，最小数据块的大小

注意: 根据：https://github.com/PaddlePaddle/Paddle/issues/8638#issuecomment-369912156 , 当数据块大小超过2MB时，我们可以有效地使用带宽。如果你想更改它，请详细查看slice_variable函数。

**代码示例**

.. code-block:: python

        from paddle.fluid.transpiler.ps_dispatcher import RoundRobin
        import paddle.fluid as fluid

        config = fluid.DistributeTranspilerConfig()
        config.slice_var_up = True
        config.split_method = RoundRobin
        config.min_block_size = 81920



