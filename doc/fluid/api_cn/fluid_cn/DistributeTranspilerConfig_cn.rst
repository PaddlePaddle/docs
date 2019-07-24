.. _cn_api_fluid_DistributeTranspilerConfig:

DistributeTranspilerConfig
-------------------------------

.. py:class:: paddle.fluid.DistributeTranspilerConfig


.. py:attribute:: slice_var_up (bool)

为多个Pserver（parameter server）将tensor切片, 默认为True。

.. py:attribute:: split_method (PSDispatcher)

可使用 RoundRobin 或者 HashName。

注意: 尝试选择最佳方法来达到Pserver间负载均衡。

.. py:attribute:: min_block_size (int)

block中分割(split)出的元素个数的最小值。

注意: 根据：`issuecomment-369912156 <https://github.com/PaddlePaddle/Paddle/issues/8638#issuecomment-369912156>`_ , 当数据块大小超过2MB时，我们可以有效地使用带宽。如果你想更改它，请详细查看 ``slice_variable`` 函数。

**代码示例**

.. code-block:: python
    
    import paddle.fluid as fluid
    config = fluid.DistributeTranspilerConfig()
    config.slice_var_up = True




