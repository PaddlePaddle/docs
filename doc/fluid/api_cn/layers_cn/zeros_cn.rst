.. _cn_api_fluid_layers_zeros:

zeros
-------------------------------

.. py:function:: paddle.fluid.layers.zeros(shape,dtype,force_cpu=False)

**zeros**

该函数创建一个张量，含有具体的维度和dtype，初始值为0.

也将stop_gradient设置为True。

参数：
    - **shape** (tuple|list|None)-输出张量的维
    - **dtype** (np.dtype|core.VarDesc.VarType|str)-输出张量的数据类型
    - **force_cpu** (bool,default False)-是否将输出保留在CPU上

返回：存储在输出中的张量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.zeros(shape=[1], dtype='int64')








