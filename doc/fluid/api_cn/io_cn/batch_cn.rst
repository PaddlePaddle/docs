.. _cn_api_fluid_io_batch:

batch
-------------------------------

.. py:function:: paddle.fluid.io.batch(reader, batch_size, drop_last=False)

该层是一个batched reader。

参数：
    - **reader** (Variable)-读取数据的数据reader
    - **batch_size** (int)-批尺寸
    - **drop_last** (bool) - 如果最后一个batch不等于batch_size，则drop最后一个batch。

返回：batched reader

返回类型：callable

              









