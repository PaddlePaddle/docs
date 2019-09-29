.. _cn_api_fluid_data:

data
-------------------------------

.. py:function:: paddle.fluid.data(name, shape, dtype='float32')

该OP会在全局block中创建变量（Variable），该全局变量可被计算图中的算子（operator）访问。该变量可作为占位符用于数据输入。例如用执行器（Executor）feed数据进该变量

注意：

  不推荐使用 ``paddle.fluid.layers.data`` ，其在之后的版本中会被删除。请使用这个 ``paddle.fluid.data`` 。 

  ``paddle.fluid.layers.data`` 在组网期间会设置创建的变量维度（shape）和数据类型（dtype），但不会检查输入数据的维度和数据类型是否符合要求。 ``paddle.fluid.data`` 会在运行过程中由Executor/ParallelExecutor检查输入数据的维度和数据类型。

参数：
    - **name** (str)- 被创建的变量的名字，具体用法请参见 :ref:`api_guide_Name` 。
    - **shape** (list|tuple)- 声明维度信息的list或tuple。
    - **dtype** (np.dtype|VarType|str)- 数据类型，支持bool，float16，float32，float64，int8，int16，int32，int64，uint8。默认值为float32。

返回：全局变量，可进行数据访问

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np

    # Creates a variable with fixed size [3, 2, 1]
    # User can only feed data of the same shape to x
    x = fluid.data(name='x', shape=[3, 2, 1], dtype='float32')

    # Creates a variable with changable batch size -1.
    # Users can feed data of any batch size into y,
    # but size of each data sample has to be [2, 1]
    y = fluid.data(name='y', shape=[-1, 2, 1], dtype='float32')

    z = x + y

    # In this example, we will feed x and y with np-ndarry "1"
    # and fetch z, like implementing "1 + 1 = 2" in PaddlePaddle
    feed_data = np.ones(shape=[3, 2, 1], dtype=np.float32)

    exe = fluid.Executor(fluid.CPUPlace())
    out = exe.run(fluid.default_main_program(),
                  feed={
                      'x': feed_data,
                      'y': feed_data
                  },
                  fetch_list=[z.name])

    # np-ndarray of shape=[3, 2, 1], dtype=float32, whose elements are 2
    print(out)


