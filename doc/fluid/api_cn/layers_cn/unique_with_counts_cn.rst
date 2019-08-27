.. _cn_api_fluid_layers_unique_with_counts:

unique_with_counts
-------------------------------

.. py:function:: paddle.fluid.layers.unique_with_counts(x, dtype='int32')

unique_with_count为 ``x`` 返回一个unique张量和一个指向该unique张量的索引以及 ``x`` 中unique元素的数量。

参数：
    - **x** (Variable) - 一个1维输入张量
    - **dtype** (np.dtype|core.VarDesc.VarType|str) – 索引张量的类型，int32，int64。

返回：元组(out, index, count)。 ``out`` 为 ``x`` 的指定dtype的unique张量, ``index`` 是一个指向 ``out`` 的索引张量, 用户可以通过该函数来转换原始的 ``x`` 张量的索引， ``count`` 是 ``x`` 中unique元素的数量。

返回类型：元组(tuple)

**代码示例**：

.. code-block:: python

    import numpy as np
    import paddle.fluid as fluid
    x = fluid.assign(np.array([2, 3, 3, 1, 5, 3], dtype='int32'))
    out, index, count = fluid.layers.unique_with_counts(x) # out is [2, 3, 1, 5];
                                               # index is [0, 1, 1, 2, 3, 1];
                                               # count is [1, 3, 1, 1]










