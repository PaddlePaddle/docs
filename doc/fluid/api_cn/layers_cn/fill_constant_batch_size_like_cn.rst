.. _cn_api_fluid_layers_fill_constant_batch_size_like:

fill_constant_batch_size_like
-------------------------------

.. py:function:: paddle.fluid.layers.fill_constant_batch_size_like(input,shape,dtype,value,input_dim_idx=0,output_dim_idx=0)

该功能创建一个张量，含有具体的shape,dtype和batch尺寸。并用 ``Value`` 中提供的常量初始化该张量。该批尺寸从输入张量中获取。它还将stop_gradient设置为True.

参数：
    - **input** (Variable)-张量，其第input_dim_idx维可指定batch_size
    - **shape** (INTS)-输出的形状
    - **dtype** (INT)-可以为numpy.dtype。输出数据类型。默认为float32
    - **value** (FLOAT)-默认为0.将要被填充的值
    - **input_dim_idx** (INT)-默认为0.输入批尺寸维的索引
    - **output_dim_idx** (INT)-默认为0.输出批尺寸维的索引

返回：具有特定形状和值的张量

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    like = fluid.layers.data(name='like', shape=[1], dtype='float32')
    data = fluid.layers.fill_constant_batch_size_like(
                input=like, shape=[1], value=0, dtype='int64')










