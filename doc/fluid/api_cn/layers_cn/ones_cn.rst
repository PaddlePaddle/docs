.. _cn_api_fluid_layers_ones:

ones
-------------------------------

.. py:function:: paddle.fluid.layers.ones(shape,dtype,force_cpu=False)

**ones**

该功能创建一个张量，有具体的维度和dtype，初始值为1。

也将stop_gradient设置为True。

参数：
    - **shape** (tuple|list)-输出张量的维
    - **dtype** (np.dtype|core.VarDesc.VarType|str)-输出张量的数据类型

返回：存储在输出中的张量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.ones(shape=[1], dtype='int64')



