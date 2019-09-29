.. _cn_api_fluid_data_feeder_DataFeeder:

DataFeeder
-------------------------------

.. py:class:: paddle.fluid.data_feeder.DataFeeder(feed_list, place, program=None)


``DataFeeder`` 负责将reader(数据读取函数)返回的数据转换为可以被 ``Executor`` 和 ``ParallelExecutor`` 解析的数据结构 - 用作executor的feed参数
reader通常是一个用来从文件中读取/生成数据样本的自定义python生成器，该函数通常会返回一个包含样本实例数据的列表。

参数：
    - **feed_list** (list) – 向模型输入的变量列表或者变量名列表
    - **place** (:ref:`cn_api_fluid_CPUPlace` | :ref:`cn_api_fluid_CUDAPlace` ) – place表明是向GPU还是CPU中输入数据。如果想向GPU中输入数据, 请使用 ``fluid.CUDAPlace(i)`` (i 代表 the GPU id)；如果向CPU中输入数据, 请使用  ``fluid.CPUPlace()``
    - **program** (:ref:`cn_api_fluid_Program` ) – 需要向其中输入数据的Program。如果为None, 会默认使用 ``default_main_program()`` 。 缺省值为None

异常情况:   ``ValueError``  – 如果一些变量不在此 Program 中


**代码示例**

.. code-block:: python

    import numpy as np
    import paddle
    import paddle.fluid as fluid
    
    place = fluid.CPUPlace()
    def reader():
        for _ in range(4):
            yield np.random.random([4]).astype('float32'), np.random.random([3]).astype('float32'),
    
    main_program = fluid.Program()
    startup_program = fluid.Program()
    
    with fluid.program_guard(main_program, startup_program):
        data_1 = fluid.layers.data(name='data_1', shape=[-1, 2, 2])
        data_2 = fluid.layers.data(name='data_2', shape=[-1, 1, 3])
        out = fluid.layers.fc(input=[data_1, data_2], size=2)
        # ...
    feeder = fluid.DataFeeder([data_1, data_2], place)
    
    exe = fluid.Executor(place)
    exe.run(startup_program)
    
    feed_data = feeder.feed(reader())
    
    # print feed_data to view feed results
    # print(feed_data['data_1'])
    # print(feed_data['data_2'])
    
    outs = exe.run(program=main_program,
                    feed=feed_data,
                    fetch_list=[out])
    print(outs)


.. py:method:: feed(iterable)


根据创建DataFeeder时候传入的feed_list(变量列表) 和iterable (自定义python生成器) 将原始数据转换为tensor结构

参数:
    - **iterable** (generator) – 自定义的python生成器，用来获取原始输入数据

返回：以变量名为key，tensor为value的dict

返回类型: dict

**代码示例**

.. code-block:: python

    # 本示例中，reader函数会返回一个长度为3的数组，每个元素都是ndarray类型，
    # 分别对应data_1, data_2, data_3的原始数据

    # feed函数内部会将每个传入的ndarray转换为Paddle内部用于计算的tensor结构
    # 返回的结果是一个size为3的dict，key分别为data_1, data_2, data_3
    # result['data_1']  为一个shape 为 [5, 2, 1, 3] 的LoD-tensor  其中5为batch size, [2, 1, 3]为data_1的shape
    # result['data_2'], result['data_3']以此类推

    import numpy as np
    import paddle.fluid as fluid
    
    def reader(limit=5):
        for i in range(1, limit + 1):
            yield np.ones([6]).astype('float32') * i , np.ones([1]).astype('int64') * i, np.random.random([9]).astype('float32')
    
    data_1 = fluid.layers.data(name='data_1', shape=[2, 1, 3])
    data_2 = fluid.layers.data(name='data_2', shape=[1], dtype='int64')
    data_3 = fluid.layers.data(name='data_3', shape=[3, 3], dtype='float32')
    feeder = fluid.DataFeeder(['data_1','data_2', 'data_3'], fluid.CPUPlace())
    
    
    result = feeder.feed(reader())
    print(result['data_1'])
    print(result['data_2'])
    print(result['data_3'])


.. py:method:: feed_parallel(iterable, num_places=None)

功能类似于 ``feed`` 函数，feed_parallel用于使用多个设备(CPU|GPU)的情况，iterable为自定义的生成器列表，
列表中的每个生成器返回的数据最后会feed到相对应的设备中


参数:
    - **iterable** (list(generator)) – 自定义的python生成器列表，列表元素个数与num_places保持一致
    - **num_places** (int) – 设备数目。默认为None。

返回: 返回值为dict的生成器，生成器返回一个键值对为 ``变量名-tensor`` 组成的dict

返回类型: generator

.. note::
   设备(CPU或GPU)的数目 - ``num-places`` 必须等于 ``iterable`` 参数中的生成器数量

**代码示例**

.. code-block:: python

    import numpy as np
    import paddle.fluid as fluid
    
    def generate_reader(batch_size, base=0, factor=1):
        def _reader():
            for i in range(batch_size):
                yield np.ones([4]) * factor + base, np.ones([4]) * factor + base + 5
        return _reader()
    
    x = fluid.layers.data(name='x', shape=[-1, 2, 2])
    y = fluid.layers.data(name='y', shape=[-1, 2, 2], dtype='float32')
    
    z = fluid.layers.elementwise_add(x, y)
    
    feeder = fluid.DataFeeder(['x','y'], fluid.CPUPlace())
    place_num = 2
    places = [fluid.CPUPlace() for x in range(place_num)]
    data = []
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    program = fluid.CompiledProgram(fluid.default_main_program()).with_data_parallel(places=places)
    
    # 打印feed_parallel结果示例
    # for item in list(feeder.feed_parallel([generate_reader(5, 0, 1), generate_reader(3, 10, 2)], 2)):
    #     print(item['x'])
    #     print(item['y'])
    
    reader_list = [generate_reader(5, 0, 1), generate_reader(3, 10, 2)]
    res = exe.run(program=program, feed=list(feeder.feed_parallel(reader_list, 2)), fetch_list=[z])
    print(res)


.. py:method::  decorate_reader(reader, multi_devices, num_places=None, drop_last=True)


将reader返回的输入数据batch转换为多个mini-batch，之后每个mini-batch都会被输入进各个设备（CPU或GPU）中。
    
参数：
        - **reader** (generator) – 一个用来自定义返回mini-batch的生成器，一个返回实例数据的reader可视为一个mini-batch(如下面例子中的 ``_mini_batch`` )
        - **multi_devices** (bool) – 指明是否使用多个设备
        - **num_places** (int，可选) – 如果 ``multi_devices`` 为 ``True`` , 可以使用此参数来设置设备数目。如果 ``num_places`` 为 ``None`` ，该函数默认使用当前训练机所有设备。默认为None。
        - **drop_last** (bool, 可选) – 如果最后一组数据的数量比设备数要小，则可使用该参数来指明是否选择丢弃最后一个组数据。 默认为 ``True``

返回：一个装饰之后的生成器，该生成器会返回匹配num_places数量的tensor数据列表

返回类型：generator

异常情况： ValueError – 如果 ``drop_last`` 值为False并且最后一组数据的minibatch数目与设备数目不相等时，产生此异常

**代码示例**

.. code-block:: python

    import numpy as np
    import paddle
    import paddle.fluid as fluid
    import paddle.fluid.compiler as compiler

    def reader():
        def _mini_batch(batch_size):
            for i in range(batch_size):
                yield np.random.random([16]).astype('float32'), np.random.randint(10, size=[1])

        for _ in range(10):
            yield _mini_batch(np.random.randint(1, 10))

    place_num = 3
    places = [fluid.CPUPlace() for _ in range(place_num)]
    data = fluid.layers.data(name='data', shape=[-1, 4, 4], dtype='float32')
    label = fluid.layers.data(name='label', shape=[-1, 1], dtype='int64')

    hidden = fluid.layers.fc(input=data, size=10)

    feeder = fluid.DataFeeder(place=places[0], feed_list=[data, label])
    reader = feeder.decorate_reader(reader, multi_devices=True, num_places=3, drop_last=True)

    exe = fluid.Executor(places[0])
    exe.run(fluid.default_startup_program())
    compiled_prog = compiler.CompiledProgram(
             fluid.default_main_program()).with_data_parallel(places=places)
    for i,data in enumerate(reader()):
        ret = exe.run(compiled_prog, feed=data, fetch_list=[hidden])
        print(ret)

