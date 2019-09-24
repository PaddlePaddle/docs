.. _cn_api_fluid_layers_fill_constant_batch_size_like:

fill_constant_batch_size_like
-------------------------------

.. py:function:: paddle.fluid.layers.fill_constant_batch_size_like(input,shape,dtype,value,input_dim_idx=0,output_dim_idx=0)

该OP创建一个形状为shape并且数据类型为dtype的Tensor，同时用 ``value`` 中提供的常量初始化该Tensor。在输入为LoDTensor并且input_dim_idx为0的
时候将输出output_dim_idx维度的大小设置为input输入的batch_size的值，创建的Tensor的stop_gradient属性默认为False。

参数：
    - **input** (Variable)- 输入的Tensor或者LoDTensor，支持数据类型为 float32， float64， int32， int64。
    - **shape** (list)- 创建Tensor的shape，最后创建的LoDTensor的shape可能会依据input发生变动。
    - **dtype** (np.dtype|core.VarDesc.VarType|str)- 创建Tensor的数据类型，支持数据类型为 float32， float64， int32， int64。
    - **value** (float|int)-  用于初始化输出Tensor的常量数据的值。
    - **input_dim_idx** (int)- 当值为0并且输入为LoDTensor的时候，创建Tensor的output_dim_idx维度会设置为input的batch_size值，默认值为0。
    - **output_dim_idx** (int) -用于指定创建的Tensor哪个维度设置为输入batch_size的值，默认值为0。

返回：创建的Tensor, 数据类型为dtype。

返回类型：(Variable)

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    like = fluid.layers.fill_constant(shape=[1,2], value=10, dtype='int64') #like=[[10, 10]]
    data = fluid.layers.fill_constant_batch_size_like(
                input=like, shape=[1], value=0, dtype='int64') #like=[[10, 10]] data=[0]