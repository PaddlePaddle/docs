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

**代码示例**

.. code-block:: python

        config = fluid.DistributeTranspilerConfig()
        config.slice_var_up = True



