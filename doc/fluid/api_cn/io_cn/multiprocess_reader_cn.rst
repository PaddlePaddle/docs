.. _cn_api_fluid_io_multiprocess_reader:

multiprocess_reader
-------------------------------

.. py:function:: paddle.fluid.io.multiprocess_reader(readers, use_pipe=True, queue_size=1000)




使用python多进程从 ``readers`` 中读取数据，然后使用 ``multiprocessing.Pipe`` 或 ``multiprocessing.Queue`` 合并所有数据。 ``readers`` 列表中的每个reader会被创建一个独立的进程来调用，reader之间应该相互独立，互不影响，避免出现多进程读取的冲突问题.

multiprocess.queue需要/dev/shm的rw访问权限，某些平台不支持。

参数：
    - **readers** (list(generator)|tuple(generator)) - python生成器list, 用来读取数据
    - **use_pipe** (bool，可选) - use_pipe控制multiprocess_reader内部用 ``pipe`` 还是 ``queue`` 来实现进程间通信，默认为 ``True`` 使用 ``pipe`` 进行通信
    - **queue_size** (int，可选) - 如果使用queue来进行进程间通信 (``use_pipe=False``), 则该参数用来设定队列大小

返回：使用多进程封装readers之后的reader

返回类型：python生成器


**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    from paddle.fluid.io import multiprocess_reader
    import numpy as np
    
    
    def fake_reader(start, end):
        def __impl__():
            for i in range(start, end):
                yield [np.array([1, 2, 3]) * i],
        return __impl__
    
    
    with fluid.program_guard(fluid.Program(), fluid.Program()):
        place = fluid.CPUPlace()
        image = fluid.layers.data(
            name='image', dtype='int64', shape=[3])
        fluid.layers.Print(image)
        reader = fluid.io.PyReader(
            feed_list=[image], capacity=2)
        image_p_1 = image + 1
        decorated_reader = multiprocess_reader(
            [fake_reader(1, 5), fake_reader(6, 10)], False)
    
        reader.decorate_sample_generator(decorated_reader, batch_size=2, places=[place])
    
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
    
        for data in reader():
            exe.run(feed=data, fetch_list=[image_p_1])

