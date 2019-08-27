.. _cn_api_fluid_io_multiprocess_reader:

multiprocess_reader
-------------------------------

.. py:function:: paddle.fluid.io.multiprocess_reader(readers, use_pipe=True, queue_size=1000)

多进程reader使用python多进程从reader中读取数据，然后使用multi process.queue或multi process.pipe合并所有数据。进程号等于输入reader的编号，每个进程调用一个reader。

multiprocess.queue需要/dev/shm的rw访问权限，某些平台不支持。

您需要首先创建多个reader，这些reader应该相互独立，这样每个进程都可以独立工作。

**代码示例**

..  code-block:: python

    reader0 = reader(["file01", "file02"])
    reader1 = reader(["file11", "file12"])
    reader1 = reader(["file21", "file22"])
    reader = multiprocess_reader([reader0, reader1, reader2],
        queue_size=100, use_pipe=False)