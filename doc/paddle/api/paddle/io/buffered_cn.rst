.. _cn_api_io_buffered:

buffered
-------------------------------

.. py:function:: paddle.io.buffered(reader, size)




创建一个缓存数据读取器，它读取数据并且存储进缓存区，从缓存区读取数据将会加速，只要缓存不是空的。

参数:
    - **reader** (callable) – 读取数据的reader
    - **size** (int) – 最大buffer的大小

返回:缓存的reader（读取器）

**代码示例**

..  code-block:: python

    import paddle

    def reader():
        for i in range(3):
            yield i

    # Create a buffered reader, and the buffer size is 2.
    buffered_reader = paddle.io.buffered(reader, 2)

    # Output: 0 1 2
    for i in buffered_reader():
        print(i)
