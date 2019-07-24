.. _cn_api_paddle_data_reader_datafeeder:

DataFeeder
-----------------------------------

.. py:class:: paddle.fluid.data_feeder.DataFeeder(feed_list, place, program=None)


DataFeeder将reader返回的数据转换为可以输入Executor和ParallelExecutor的数据结构。reader通常返回一个小批量数据条目列表。列表中的每个数据条目都是一个样本。每个样本都是具有一个或多个特征的列表或元组。

简单用法如下：

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    place = fluid.CPUPlace()
    img = fluid.layers.data(name='image', shape=[1, 28, 28])
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    feeder = fluid.DataFeeder([img, label], fluid.CPUPlace())
    result = feeder.feed([([0] * 784, [9]), ([1] * 784, [1])])


如果您想在使用多个GPU训练模型时预先将数据单独输入GPU端，可以使用decorate_reader函数。


**代码示例**

..  code-block:: python

    import paddle
    import paddle.fluid as fluid
    
    place=fluid.CUDAPlace(0)
    data = fluid.layers.data(name='data', shape=[3, 224, 224], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    
    feeder = fluid.DataFeeder(place=place, feed_list=[data, label])
    reader = feeder.decorate_reader(
        paddle.batch(paddle.dataset.flowers.train(), batch_size=16), multi_devices=False)


参数：
    - **feed_list**  (list) –  将输入模型的变量或变量的名称。
    - **place**  (Place) – place表示将数据输入CPU或GPU，如果要将数据输入GPU，请使用fluid.CUDAPlace(i)（i表示GPU的ID），如果要将数据输入CPU，请使用fluid.CPUPlace()。
    - **program**  (Program) –将数据输入的Program，如果Program为None，它将使用default_main_program() 。默认值None。

抛出异常：     ``ValueError`` – 如果某些变量未在Program中出现


**代码示例**

..  code-block:: python

    import numpy as np
    import paddle
    import paddle.fluid as fluid

    place = fluid.CPUPlace()

    def reader():
        yield [np.random.random([4]).astype('float32'), np.random.random([3]).astype('float32')],

    main_program = fluid.Program()
    startup_program = fluid.Program()

    with fluid.program_guard(main_program, startup_program):
        data_1 = fluid.layers.data(name='data_1', shape=[1, 2, 2])
        data_2 = fluid.layers.data(name='data_2', shape=[1, 1, 3])
        out = fluid.layers.fc(input=[data_1, data_2], size=2)
        # ...

    feeder = fluid.DataFeeder([data_1, data_2], place)

    exe = fluid.Executor(place)
    exe.run(startup_program)
    for data in reader():
        outs = exe.run(program=main_program,
                       feed=feeder.feed(data),
                       fetch_list=[out])


.. py:method::  feed(iterable)

根据feed_list和iterable，将输入转换成一个数据结构，该数据结构可以输入Executor和ParallelExecutor。

参数：
    - **iterable** (list|tuple) – 输入的数据

返回： 转换结果

返回类型： dict

**代码示例**

..  code-block:: python

        import numpy.random as random
        import paddle.fluid as fluid
        
        def reader(limit=5):
            for i in range(limit):
                    yield random.random([784]).astype('float32'), random.random([1]).astype('int64'), random.random([256]).astype('float32')
        
        data_1 = fluid.layers.data(name='data_1', shape=[1, 28, 28])
        data_2 = fluid.layers.data(name='data_2', shape=[1], dtype='int64')
        data_3 = fluid.layers.data(name='data_3', shape=[16, 16], dtype='float32')
        feeder = fluid.DataFeeder(['data_1','data_2', 'data_3'], fluid.CPUPlace())
        
        result = feeder.feed(reader())



.. py:method::  feed_parallel(iterable, num_places=None)

需要多个mini-batches。每个mini-batch都将提前在每个设备上输入。

参数：
    - **iterable** (list|tuple) – 输入的数据。
    - **num_places**  (int) – 设备编号，默认值为None。

返回： 转换结果

返回类型： dict



.. note::

    设备数量和mini-batches数量必须一致。

**代码示例**

..  code-block:: python

        import numpy.random as random
        import paddle.fluid as fluid
        
        def reader(limit=10):
            for i in range(limit):
                yield [random.random([784]).astype('float32'), random.randint(10)],
        
        x = fluid.layers.data(name='x', shape=[1, 28, 28])
        y = fluid.layers.data(name='y', shape=[1], dtype='int64')
        
        feeder = fluid.DataFeeder(['x','y'], fluid.CPUPlace())
        place_num = 2
        places = [fluid.CPUPlace() for x in range(place_num)]
        data = []
        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())
        program = fluid.CompiledProgram(fluid.default_main_program()).with_data_parallel(places=places)
        for item in reader():
            data.append(item)
            if place_num == len(data):
                exe.run(program=program, feed=list(feeder.feed_parallel(data, place_num)), fetch_list=[])
                data = []


.. py:method::  decorate_reader(reader, multi_devices, num_places=None, drop_last=True)

将输入数据转换成reader返回的多个mini-batches。每个mini-batch分别送入各设备中。

参数：
    - **reader** (function) – reader是可以生成数据的函数。
    - **multi_devices** (bool) – 是否用多个设备。
    - **num_places** (int) – 如果multi_devices是True, 你可以指定GPU的使用数量, 如果multi_devices是None, 会使用当前机器的所有GPU ，默认值None。
    - **drop_last** (bool) – 如果最后一个batch的大小小于batch_size，选择是否删除最后一个batch，默认值True。

返回： 转换结果

返回类型： dict

抛出异常：     ``ValueError`` – 如果drop_last为False并且数据batch和设备数目不匹配。

**代码示例**

..  code-block:: python

        import numpy.random as random
        import paddle
        import paddle.fluid as fluid
        
        def reader(limit=5):
            for i in range(limit):
                yield (random.random([784]).astype('float32'), random.random([1]).astype('int64')),
        
        place=fluid.CUDAPlace(0)
        data = fluid.layers.data(name='data', shape=[1, 28, 28], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        
        feeder = fluid.DataFeeder(place=place, feed_list=[data, label])
        reader = feeder.decorate_reader(reader, multi_devices=False)
        
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        for data in reader():
            exe.run(feed=data)