.. _cn_api_io_cache:

cache
-------------------------------

.. py:function:: paddle.io.cache(reader)




缓存reader数据到内存中，小心此方法可能会花长时间来处理数据，并且会占用大量内存。 ``reader()`` 只能被调用一次。

参数:
    - **reader** (callable) – 读取数据的reader，每次都会yields数据。

返回：每次都会从内存中yields数据的一个装饰reader。

返回类型：数据保存在内存的reader（读取器）

**代码示例**

..  code-block:: python

    import paddle

    def reader():
        for i in range(3):
            yield i

    # All data is cached into memory
    cached_reader = paddle.io.cache(reader)

    # Output: 0 1 2
    for i in cached_reader():
        print(i)
