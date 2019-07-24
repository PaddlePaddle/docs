.. _cn_api_fluid_layers_fill_constant:

fill_constant
-------------------------------

.. py:function:: paddle.fluid.layers.fill_constant(shape,dtype,value,force_cpu=False,out=None)

该功能创建一个张量，含有具体的shape,dtype和batch尺寸。并用 ``value`` 中提供的常量初始化该张量。

创建张量的属性stop_gradient设为True。

参数：
    - **shape** (tuple|list|None)-输出张量的形状
    - **dtype** (np.dtype|core.VarDesc.VarType|str)-输出张量的数据类型
    - **value** (float)-用于初始化输出张量的常量值
    - **out** (Variable)-输出张量
    - **force_cpu** (True|False)-若设为true,数据必须在CPU上

返回：存储着输出的张量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.fill_constant(shape=[1], value=0, dtype='int64')









