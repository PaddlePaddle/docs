.. _cn_api_fluid_io_xmap_readers:

xmap_readers
-------------------------------

.. py:function:: paddle.fluid.io.xmap_readers(mapper, reader, process_num, buffer_size, order=False)

多线程下，使用自定义映射器 reader 返回样本到输出队列。

参数：
    - **mapper** （callable） - 映射 reader 数据的函数。
    - **reader** （callable） - 产生数据的 reader。
    - **process_num** （int） - 处理样本的线程数。
    - **buffer_size** （int） - 数据缓冲队列大小。
    - **order** （bool） - 是否保持原始 reader 数据顺序，默认为 False。

返回：一个用户定义的 reader `装饰器 <https://en.wikipedia.org/wiki/Python_syntax_and_semantics#Decorators>`_ 。

返回类型：callable，可调用对象。

**代码示例**：

.. code-block:: python

    import paddle.reader as reader
    import time

    def reader_creator_10(dur):
        def reader():
            for i in range(10):
                time.sleep(dur)
                yield i
        return reader

    def mapper(x):
        return (x + 1)

    orders = (True, False)
    thread_num = (1, 2, 4, 8, 16)
    buffer_size = (1, 2, 4, 8, 16)
    for order in orders:
        for t_num in thread_num:
            for size in buffer_size:
                user_reader = reader.xmap_readers(mapper,
                                                  reader_creator_10(0),
                                                  t_num, size, order)
                for n in range(3):
                    result = list()
                    for i in user_reader():
                        result.append(i)
                    if not order:
                        result.sort()
                    for idx, e in enumerate(result):
                        assert e == mapper(idx)