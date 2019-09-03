.. _cn_api_fluid_io_cache:

cache
-------------------------------

.. py:function:: paddle.fluid.io.cache(reader)

缓存reader数据到内存中，小心此方法可能会花长时间来处理数据，并且会占用大量内存。 ``reader()`` 只能被调用一次。

参数:
    - **reader** (callable) – 读取数据的reader，每次都会yields数据。

返回：每次都会从内存中yields数据的一个装饰reader。

返回类型：生成器