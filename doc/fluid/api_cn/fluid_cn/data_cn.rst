.. _cn_api_fluid_data:

data
-------------------------------

.. py:function:: paddle.fluid.data(name, shape, dtype='float32', type=VarType.LOD_TENSOR)

该OP会在全局scope中创建变量（Variable），该全局变量可被计算图中的算子（operator）访问。

注意：

  不推荐使用 ``paddle.fluid.layers.data`` ，其在之后的版本中会被删除。请使用这个 ``paddle.fluid.data`` 。 

  ``paddle.fluid.layers.data`` 在组网期间会设置创建的变量shape，但不会检查输入数据的shape是否与符合要求。 ``paddle.fluid.data`` 会在运行过程中由Executor/ParallelExecutor检查输入数据的维度。

参数：
    - **name** (str)- 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。
    - **shape** (list|tuple)- 声明维度信息的list或tuple。
    - **dtype** (np.dtype|VarType|str)- 数据类型，支持bool，float16，float32，float64，int8，int16，int32，int64，uint8。
    - **type** (VarType)- 输出类型，支持VarType.LOD_TENSOR，VarType.SELECTED_ROWS，VarType.NCCL_ID。默认为VarType.LOD_TENSOR。

返回：全局变量，可进行数据访问

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    # Creates a variable with fixed size [1, 2, 3]
    # User can only feed data of the same shape to x
    x = fluid.data(name='x', shape=[1, 2, 3], dtype='int64')

    # Creates a variable with changable batch size -1.
    # Users can feed data of any batch size into y, 
    # but size of each data sample has to be [3, 224, 224]
    y = fluid.data(name='y', shape=[-1, 3, 224, 224], dtype='float32')


