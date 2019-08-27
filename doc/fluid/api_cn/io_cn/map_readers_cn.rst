.. _cn_api_fluid_io_map_readers:

map_readers
-------------------------------

.. py:function::   paddle.fluid.io.map_readers(func, *readers)

创建使用每个数据读取器的输出作为参数输出函数返回值的数据读取器。

参数：
    - **func**  - 使用的函数. 函数类型应为(Sample) => Sample
    - **readers**  - 其输出将用作func参数的reader。

类型：callable

返回： 被创建数据的读取器





