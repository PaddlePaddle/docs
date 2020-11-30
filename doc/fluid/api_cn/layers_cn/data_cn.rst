.. _cn_api_fluid_layers_data:

data
-------------------------------


.. py:function:: paddle.fluid.layers.data(name, shape, append_batch_size=True, dtype='float32', lod_level=0, type=VarType.LOD_TENSOR, stop_gradient=True)




该OP会在全局block中创建变量（Variable），该全局变量可被计算图中的算子（operator）访问。

注意：

  不推荐使用 ``paddle.fluid.layers.data`` ，因其在之后的版本中会被删除。请使用 ``paddle.fluid.data`` 。 

  ``paddle.fluid.layers.data`` 在组网期间会设置创建的变量维度（shape）和数据类型（dtype），但不会检查输入数据的维度和数据类型是否符合要求。 ``paddle.fluid.data`` 会在运行过程中由Executor/ParallelExecutor检查输入数据的维度。

  如果想输入变长输入，用户可以直接输入这个 ``paddle.fluid.layers.data`` 且PaddlePaddle会按具体输入的形状运行。或者也可以使用 ``paddle.fluid.data`` 时将变长维度设为-1。

  本API创建的变量默认 ``stop_gradient`` 属性为true，这意味这反向梯度不会被传递过这个数据变量。如果用户想传递反向梯度，可以设置 ``var.stop_gradient = False`` 。

参数：
    - **name** (str)- 被创建的变量的名字，具体用法请参见 :ref:`api_guide_Name` 。
    - **shape** (list)- 声明维度信息的list。如果 ``append_batch_size`` 为True且内部没有维度值为-1，则应将其视为每个样本的形状。 否则，应将其视为batch数据的形状。
    - **append_batch_size** (bool)-

        1.如果为True，则在维度（shape）的开头插入-1。
        例如，如果shape=[1],则输出shape为[-1,1]。可用于设置运行期间不同batch大小。

        2.如果维度（shape）包含-1，比如shape=[-1,1]。
        append_batch_size会强制变为为False（表示无效），因为PaddlePaddle不能在shape上设置一个以上的未知数。

    - **dtype** (np.dtype|VarType|str)- 数据类型，支持bool，float16，float32，float64，int8，int16，int32，int64，uint8。
    - **type** (VarType)- 输出类型，支持VarType.LOD_TENSOR，VarType.SELECTED_ROWS，VarType.NCCL_ID。默认为VarType.LOD_TENSOR。
    - **lod_level** (int)- LoD层。0表示输入数据不是一个序列。默认值为0。
    - **stop_gradient** (bool)- 提示是否应该停止计算梯度，默认值为True。

返回：全局变量，可进行数据访问

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(name='x', shape=[784], dtype='float32')










