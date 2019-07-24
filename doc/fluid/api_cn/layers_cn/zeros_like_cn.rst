.. _cn_api_fluid_layers_zeros_like:

zeros_like
-------------------------------

.. py:function:: paddle.fluid.layers.zeros_like(x, out=None)

**zeros_like**

该函数创建一个和x具有相同的形状和数据类型的全零张量

参数：
    - **x** (Variable)-指定形状和数据类型的输入张量
    - **out** (Variable)-输出张量
    
返回：存储输出的张量变量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name='x', dtype='float32', shape=[3], append_batch_size=False)
    data = fluid.layers.zeros_like(x) # [0.0, 0.0, 0.0]






