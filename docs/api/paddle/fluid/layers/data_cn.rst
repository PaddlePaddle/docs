.. _cn_api_fluid_data:

data
-------------------------------


.. py:function:: paddle.fluid.data(name, shape, dtype='float32', lod_level=0)




该OP会在全局block中创建变量（Variable），该全局变量可被计算图中的算子（operator）访问。该变量可作为占位符用于数据输入。例如用执行器（Executor）feed数据进该变量

注意：

  不推荐使用 ``paddle.fluid.layers.data``，其在之后的版本中会被删除。请使用这个 ``paddle.fluid.data`` 。 

  ``paddle.fluid.layers.data`` 在组网期间会设置创建的变量维度（shape）和数据类型（dtype），但不会检查输入数据的维度和数据类型是否符合要求。``paddle.fluid.data`` 会在运行过程中由Executor/ParallelExecutor检查输入数据的维度和数据类型。

  如果想输入变长输入，可以使用 ``paddle.fluid.data`` 时将变长维度设为-1，或者直接输入 ``paddle.fluid.layers.data`` 且PaddlePaddle会按具体输入的形状运行。

  本API创建的变量默认 ``stop_gradient`` 属性为true，这意味这反向梯度不会被传递过这个数据变量。如果用户想传递反向梯度，可以设置 ``var.stop_gradient = False`` 。

参数
::::::::::::

    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
    - **shape** (list|tuple)- 声明维度信息的list或tuple。
    - **dtype** (np.dtype|VarType|str，可选)- 数据类型，支持bool，float16，float32，float64，int8，int16，int32，int64，uint8。默认值为float32。
    - **lod_level** (int，可选)- LoDTensor变量的LoD level数，LoD level是PaddlePaddle的高级特性，一般任务中不会需要更改此默认值，关于LoD level的详细适用场景和用法请见 :ref:`cn_user_guide_lod_tensor`。默认值为0。

返回
::::::::::::
全局变量，可进行数据访问

返回类型
::::::::::::
Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.data