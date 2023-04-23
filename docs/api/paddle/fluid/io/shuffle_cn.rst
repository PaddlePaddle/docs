.. _cn_api_fluid_io_shuffle:

shuffle
-------------------------------

.. py:function:: paddle.fluid.io.shuffle(reader, buffer_size)




该接口创建一个数据读取器，其功能是将原始数据读取器的数据打乱，然后返回无序的数据。

从原始数据读取器取出buf_size个数据到缓冲区，将缓冲区数据打乱，然后将无序的数据依次返回。当缓冲区数据全部输出后，再次执行上述步骤。

参数
::::::::::::

    - **reader** (callable)  – 原始数据读取器。
    - **buf_size** (int)  – 缓冲区保存数据的个数。

返回
::::::::::::
 返回无序数据的数据读取器

返回类型
::::::::::::
 callable

..  code-block:: python

    import paddle.fluid as fluid
    def reader():
        for i in range(5):
            yield i
    shuffled_reader = fluid.io.shuffle(reader, 3)
    for e in shuffled_reader():
        print(e)
    # 输出结果是0~4的无序排列
