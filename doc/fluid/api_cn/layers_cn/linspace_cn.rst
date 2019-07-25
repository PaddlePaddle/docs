.. _cn_api_fluid_layers_linspace:

linspace
-------------------------------

.. py:function:: paddle.fluid.layers.linspace(start, stop, num, dtype)

在给定区间内返回固定数目的均匀间隔的值。
 
第一个entry是start，最后一个entry是stop。在Num为1的情况下，仅返回start。类似numpy的linspace功能。

参数：
    - **start** (float|Variable)-序列中的第一个entry。 它是一个浮点标量，或是一个数据类型为'float32'|'float64'、形状为[1]的张量。
    - **stop** (float|Variable)-序列中的最后一个entry。 它是一个浮点标量，或是一个数据类型为'float32'|'float64'、形状为[1]的张量。
    - **num** (int|Variable)-序列中的entry数。 它是一个整型标量，或是一个数据类型为int32、形状为[1]的张量。
    - **dtype** (string)-‘float32’|’float64’，输出张量的数据类型。

返回：存储一维张量的张量变量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

      import paddle.fluid as fluid
      data = fluid.layers.linspace(0, 10, 5, 'float32') # [0.0,  2.5,  5.0,  7.5, 10.0]
      data = fluid.layers.linspace(0, 10, 1, 'float32') # [0.0]





