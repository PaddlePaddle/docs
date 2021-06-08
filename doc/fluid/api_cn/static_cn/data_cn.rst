.. _cn_api_static_cn_data:

data
-------------------------------


.. py:function:: paddle.static.data(name, shape, dtype=None, lod_level=0)




该OP会在全局block中创建变量（Variable），该全局变量可被计算图中的算子（operator）访问。该变量可作为占位符用于数据输入。例如用执行器（Executor）feed数据进该变量,当 ``dtype`` 为None时， ``dtype`` 将通过 ``padle.get_default_dtype()`` 获取全局类型。


参数：
    - **name** (str)- 被创建的变量的名字，具体用法请参见 :ref:`api_guide_Name` 。
    - **shape** (list|tuple)- 声明维度信息的list或tuple。可以在某个维度上设置None或-1，以指示该维度可以是任何大小。例如，将可变batchsize设置为None或-1。
    - **dtype** (np.dtype|str，可选)- 数据类型，支持bool，float16，float32，float64，int8，int16，int32，int64，uint8。默认值为None。当 ``dtype`` 为None时， ``dtype`` 将通过 ``padle.get_default_dtype()`` 获取全局类型。
    - **lod_level** (int，可选)- LoDTensor变量的LoD level数，LoD level是PaddlePaddle的高级特性，一般任务中不会需要更改此默认值，关于LoD level的详细适用场景和用法请见 :ref:`cn_user_guide_lod_tensor` 。默认值为0。

返回：全局变量，可进行数据访问

返回类型：Variable

**代码示例**：

.. code-block:: python

    import numpy as np
    import paddle.fluid as fluid
    import paddle
    paddle.enable_static()
    # Creates a variable with fixed size [3, 2, 1]
    # User can only feed data of the same shape to x
    # the dtype is not set, so it will set "float32" by
    # paddle.get_default_dtype(). You can use paddle.get_default_dtype() to 
    # change the global dtype
    x = paddle.static.data(name='x', shape=[3, 2, 1])
    # Creates a variable with changeable batch size -1.
    # Users can feed data of any batch size into y,
    # but size of each data sample has to be [2, 1]
    y = paddle.static.data(name='y', shape=[-1, 2, 1], dtype='float32')
    z = x + y
    # In this example, we will feed x and y with np-ndarray "1"
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
