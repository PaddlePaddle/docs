.. _cn_api_fluid_layers_cast:

cast
-------------------------------

.. py:function:: paddle.fluid.layers.cast(x,dtype)

该层传入变量x, 并用x.dtype将x转换成dtype类型，作为输出。如果输出的dtype和输入的dtype相同，则使用cast是没有意义的，但如果真的这么做了也不会报错。

参数：
    - **x** (Variable)-转换函数的输入变量
    - **dtype** (np.dtype|core.VarDesc.VarType|str)-输出变量的数据类型

返回：转换后的输出变量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(name='x', shape=[13], dtype='float32')
    result = fluid.layers.cast(x=data, dtype='float64')









