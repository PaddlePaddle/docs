.. _cn_api_fluid_layers_data:

data
-------------------------------

.. py:function:: paddle.fluid.layers.data(name, shape, append_batch_size=True, dtype='float32', lod_level=0, type=VarType.LOD_TENSOR, stop_gradient=True)

该OP会生成可被计算图中的所有算子（operator）访问的全局变量。

注意：

  不推荐使用 ``paddle.fluid.layers.data`` ，因其在之后的版本中会被删除。请使用 ``paddle.fluid.data`` 。 

  ``paddle.fluid.layers.data`` 在编译期间会使用shape推断网络中变量的维度，但不会检查输入数据的shape是否与符合要求。 ``paddle.fluid.data`` 会在运行过程中由Executor/ParallelExecutor检查输入数据的维度。

参数：
    - **name** (str)- 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。
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










