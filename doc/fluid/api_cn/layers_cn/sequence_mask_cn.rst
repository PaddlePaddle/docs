.. _cn_api_fluid_layers_sequence_mask:

sequence_mask
-------------------------------

.. py:function::  paddle.fluid.layers.sequence_mask(x, maxlen=None, dtype='int64', name=None)

该层根据输入 ``x`` 和 ``maxlen`` 输出一个掩码，数据类型为dtype。

假设x是一个形状为[d_1, d_2，…, d_n]的张量。， y是一个形为[d_1, d_2，… ，d_n, maxlen]的掩码，其中:

.. math::

  y(i_1, i_2,..., i_n, j) = (j < x(i_1, i_2,..., i_n))

参数：
  - **x** (Variable) - sequence_mask层的输入张量，其元素是小于maxlen的整数。
  - **maxlen** (int|None) - 序列的最大长度。如果maxlen为空，则用max(x)替换。
  - **dtype** (np.dtype|core.VarDesc.VarType|str) - 输出的数据类型
  - **name** (str|None) - 此层的名称(可选)。如果没有设置，该层将被自动命名。

返回： sequence mask 的输出

返回类型： Variable

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.layers as layers
    
    x = fluid.layers.data(name='x', shape=[10], dtype='float32', lod_level=1)
    mask = layers.sequence_mask(x=x)










