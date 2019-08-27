.. _cn_api_fluid_io_xmap_readers:

xmap_readers
-------------------------------

.. py:function:: paddle.fluid.io.xmap_readers(mapper, reader, process_num, buffer_size, order=False)

通过多线程方式，通过用户自定义的映射器mapper来映射reader返回的样本（到输出队列）。

参数：
    - **mapper** （callable） - 一种映射reader数据的函数。
    - **reader** （callable） - 产生数据的reader。
    - **process_num** （int） - 用于处理样本的线程数目。
    - **buffer_size** （int） - 存有待读取数据的队列的大小。
    - **order** （bool） - 是否保持原始reader的数据顺序。 默认为False。

返回：一个将原数据进行映射后的decorated reader。

返回类型： callable