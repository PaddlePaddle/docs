.. _cn_api_fluid_io_buffered:

buffered
-------------------------------

.. py:function:: paddle.fluid.io.buffered(reader, size)

创建一个缓存数据读取器，它读取数据并且存储进缓存区，从缓存区读取数据将会加速，只要缓存不是空的。

参数:
    - **reader** (callable) – 读取数据的reader
    - **size** (int) – 最大buffer的大小

返回:缓存的reader（读取器）