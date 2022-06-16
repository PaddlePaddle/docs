.. _cn_api_fluid_io_PyReader:

PyReader
-------------------------------

.. py:class:: paddle.fluid.io.PyReader(feed_list=None, capacity=None, use_double_buffer=True, iterable=True, return_list=False)





在python中为数据输入创建一个reader对象。将使用python线程预取数据，并将其异步插入队列。当调用Executor.run（…）时，将自动提取队列中的数据。

参数
::::::::::::

    - **feed_list** (list(Variable)|tuple(Variable)) - feed变量列表，由 ``fluid.layers.data()`` 创建。
    - **capacity** (int) - PyReader对象内部维护队列的容量大小。单位是batch数量。若reader读取速度较快，建议设置较大的capacity值。
    - **use_double_buffer** (bool) - 是否使用 ``double_buffer_reader``。若use_double_buffer=True，PyReader会异步地预读取下一个batch的数据，可加速数据读取过程，但同时会占用少量的CPU/GPU存储，即一个batch输入数据的存储空间。
    - **iterable** (bool) - 所创建的DataLoader对象是否可迭代。
    - **return_list** (bool) - 每个设备上的数据是否以list形式返回。仅在iterable = True模式下有效。若return_list = False，每个设备上的返回数据均是str -> LoDTensor的映射表，其中映射表的key是每个输入变量的名称。若return_list = True，则每个设备上的返回数据均是list(LoDTensor)。推荐在静态图模式下使用return_list = False，在动态图模式下使用return_list = True。


返回
::::::::::::
 被创建的reader对象

返回类型
::::::::::::
 reader (Reader)


代码示例
::::::::::::

1. 如果iterable=False，则创建的PyReader对象几乎与 ``fluid.layers.py_reader（）`` 相同。算子将被插入program中。用户应该在每个epoch之前调用 ``start（）``，并在epoch结束时捕获 ``Executor.run（）`` 抛出的 ``fluid.core.EOFException``。一旦捕获到异常，用户应该调用 ``reset（）`` 手动重置reader。

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np

    EPOCH_NUM = 3
    ITER_NUM = 5
    BATCH_SIZE = 3
    
    def network(image, label):
        # 用户定义网络，此处以softmax回归为例
        predict = fluid.layers.fc(input=image, size=10, act='softmax')
        return fluid.layers.cross_entropy(input=predict, label=label) 

    def reader_creator_random_image_and_label(height, width):
        def reader():
            for i in range(ITER_NUM):
                fake_image = np.random.uniform(low=0,
                                               high=255,
                                               size=[height, width])
                fake_label = np.ones([1])
                yield fake_image, fake_label
        return reader

    image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    reader = fluid.io.PyReader(feed_list=[image, label],
                               capacity=4,
                               iterable=False)

    user_defined_reader = reader_creator_random_image_and_label(784, 784)
    reader.decorate_sample_list_generator(
        paddle.batch(user_defined_reader, batch_size=BATCH_SIZE))
    
    loss = network(image, label)
    executor = fluid.Executor(fluid.CPUPlace())
    executor.run(fluid.default_startup_program())
    for i in range(EPOCH_NUM):
        reader.start()
        while True:
            try:
                executor.run(feed=None)
            except fluid.core.EOFException:
                reader.reset()
                break


2. 如果iterable=True，则创建的PyReader对象与程序分离。程序中不会插入任何算子。在本例中，创建的reader是一个python生成器，它是可迭代的。用户应将从PyReader对象生成的数据输入 ``Executor.run(feed=...)`` 。

.. code-block:: python

   import paddle
   import paddle.fluid as fluid
   import numpy as np

   EPOCH_NUM = 3
   ITER_NUM = 5
   BATCH_SIZE = 10

   def network(image, label):
        # 用户定义网络，此处以softmax回归为例
        predict = fluid.layers.fc(input=image, size=10, act='softmax')
        return fluid.layers.cross_entropy(input=predict, label=label)   

   def reader_creator_random_image(height, width):
       def reader():
           for i in range(ITER_NUM):
               fake_image = np.random.uniform(low=0, high=255, size=[height, width]),
               fake_label = np.ones([1])
               yield fake_image, fake_label
       return reader

   image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')
   label = fluid.layers.data(name='label', shape=[1], dtype='int64')
   reader = fluid.io.PyReader(feed_list=[image, label], capacity=4, iterable=True, return_list=False)

   user_defined_reader = reader_creator_random_image(784, 784)
   reader.decorate_sample_list_generator(
       paddle.batch(user_defined_reader, batch_size=BATCH_SIZE),
       fluid.core.CPUPlace())
   loss = network(image, label)
   executor = fluid.Executor(fluid.CPUPlace())
   executor.run(fluid.default_startup_program())

   for _ in range(EPOCH_NUM):
       for data in reader():
           executor.run(feed=data, fetch_list=[loss])

3. return_list=True，返回值将用list表示而非dict，通常用于动态图模式中。

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np

    EPOCH_NUM = 3
    ITER_NUM = 5
    BATCH_SIZE = 10

    def reader_creator_random_image(height, width):
        def reader():
            for i in range(ITER_NUM):
                yield np.random.uniform(low=0, high=255, size=[height, width]), \
                    np.random.random_integers(low=0, high=9, size=[1])
        return reader

    place = fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        py_reader = fluid.io.PyReader(capacity=2, return_list=True)
        user_defined_reader = reader_creator_random_image(784, 784)
        py_reader.decorate_sample_list_generator(
            paddle.batch(user_defined_reader, batch_size=BATCH_SIZE),
            place)
        for image, label in py_reader():
            relu = fluid.layers.relu(image)

方法
::::::::::::
start()
'''''''''

启动数据输入线程。只能在reader对象不可迭代时调用。

**代码示例**

.. code-block:: python

  import paddle
  import paddle.fluid as fluid
  import numpy as np

  BATCH_SIZE = 10
     
  def generator():
    for i in range(5):
       yield np.random.uniform(low=0, high=255, size=[784, 784]),
     
  image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')
  reader = fluid.io.PyReader(feed_list=[image], capacity=4, iterable=False)
  reader.decorate_sample_list_generator(
    paddle.batch(generator, batch_size=BATCH_SIZE))
     
  executor = fluid.Executor(fluid.CPUPlace())
  executor.run(fluid.default_startup_program())
  for i in range(3):
    reader.start()
    while True:
        try:
            executor.run(feed=None)
        except fluid.core.EOFException:
            reader.reset()
            break

reset()
'''''''''

当 ``fluid.core.EOFException`` 抛出时重置reader对象。只能在reader对象不可迭代时调用。

**代码示例**

.. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np
            
            BATCH_SIZE = 10
            
            def generator():
                for i in range(5):
                    yield np.random.uniform(low=0, high=255, size=[784, 784]),
            
            image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')
            reader = fluid.io.PyReader(feed_list=[image], capacity=4, iterable=False)
            reader.decorate_sample_list_generator(
                paddle.batch(generator, batch_size=BATCH_SIZE))
            
            executor = fluid.Executor(fluid.CPUPlace())
            executor.run(fluid.default_startup_program())
            for i in range(3):
                reader.start()
                while True:
                    try:
                        executor.run(feed=None)
                    except fluid.core.EOFException:
                        reader.reset()
                        break

decorate_sample_generator(sample_generator, batch_size, drop_last=True, places=None)
'''''''''

设置PyReader对象的数据源。

提供的 ``sample_generator`` 应该是一个python生成器，它生成的数据类型应为list(numpy.ndarray)。

当PyReader对象可迭代时，必须设置 ``places`` 。

如果所有的输入都没有LOD，这个方法比 ``decorate_sample_list_generator(paddle.batch(sample_generator, ...))`` 更快。

**参数**

  - **sample_generator** (generator)  – Python生成器，yield 类型为list(numpy.ndarray)
  - **batch_size** (int) – batch size，必须大于0
  - **drop_last** (bool) – 当样本数小于batch数量时，是否删除最后一个batch
  - **places** (None|list(CUDAPlace)|list(CPUPlace)) –  位置列表。当PyReader可迭代时必须被提供

**代码示例**

.. code-block:: python
     
            import paddle.fluid as fluid
            import numpy as np

            EPOCH_NUM = 3
            ITER_NUM = 15
            BATCH_SIZE = 3
            
            def network(image, label):
                # 用户定义网络，此处以softmax回归为例
                predict = fluid.layers.fc(input=image, size=10, act='softmax')
                return fluid.layers.cross_entropy(input=predict, label=label)    
            
            def random_image_and_label_generator(height, width):
                def generator():
                    for i in range(ITER_NUM):
                        fake_image = np.random.uniform(low=0,
                                                       high=255,
                                                       size=[height, width])
                        fake_label = np.array([1])
                        yield fake_image, fake_label
                return generator
            
            image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            reader = fluid.io.PyReader(feed_list=[image, label], capacity=4, iterable=True)
            
            user_defined_generator = random_image_and_label_generator(784, 784)
            reader.decorate_sample_generator(user_defined_generator,
                                             batch_size=BATCH_SIZE,
                                             places=[fluid.CPUPlace()])
            loss = network(image, label)
            executor = fluid.Executor(fluid.CPUPlace())
            executor.run(fluid.default_startup_program())
     
            for _ in range(EPOCH_NUM):
                for data in reader():
                    executor.run(feed=data, fetch_list=[loss])

decorate_sample_list_generator(reader, places=None)
'''''''''

设置PyReader对象的数据源。

提供的 ``reader`` 应该是一个python生成器，它生成列表（numpy.ndarray）类型的批处理数据。

当PyReader对象不可迭代时，必须设置 ``places`` 。

**参数**

  - **reader** (generator)  – 返回列表（numpy.ndarray）类型的批处理数据的Python生成器
  - **places** (None|list(CUDAPlace)|list(CPUPlace)) –  位置列表。当PyReader可迭代时必须被提供

**代码示例**

.. code-block:: python
            
            import paddle
            import paddle.fluid as fluid
            import numpy as np

            EPOCH_NUM = 3
            ITER_NUM = 15
            BATCH_SIZE = 3
            
            def network(image, label):
                # 用户定义网络，此处以softmax回归为例
                predict = fluid.layers.fc(input=image, size=10, act='softmax')
                return fluid.layers.cross_entropy(input=predict, label=label)
               
            def random_image_and_label_generator(height, width):
                def generator():
                    for i in range(ITER_NUM):
                        fake_image = np.random.uniform(low=0,
                                                       high=255,
                                                       size=[height, width])
                        fake_label = np.ones([1])
                        yield fake_image, fake_label
                return generator
            
            image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            reader = fluid.io.PyReader(feed_list=[image, label], capacity=4, iterable=True)
            
            user_defined_generator = random_image_and_label_generator(784, 784)
            reader.decorate_sample_list_generator(
                paddle.batch(user_defined_generator, batch_size=BATCH_SIZE),
                fluid.core.CPUPlace())
            loss = network(image, label)
            executor = fluid.Executor(fluid.core.CPUPlace())
            executor.run(fluid.default_startup_program())
     
            for _ in range(EPOCH_NUM):
                for data in reader():
                    executor.run(feed=data, fetch_list=[loss])

decorate_batch_generator(reader, places=None)
'''''''''

设置PyReader对象的数据源。

提供的 ``reader`` 应该是一个python生成器，它生成列表（numpy.ndarray）类型或LoDTensor类型的批处理数据。

当PyReader对象不可迭代时，必须设置 ``places`` 。

**参数**

  - **reader** (generator)  – 返回LoDTensor类型的批处理数据的Python生成器
  - **places** (None|list(CUDAPlace)|list(CPUPlace)) –  位置列表。当PyReader可迭代时必须被提供

**代码示例**

.. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            EPOCH_NUM = 3
            ITER_NUM = 15
            BATCH_SIZE = 3
            
            def network(image, label):
                # 用户定义网络，此处以softmax回归为例
                predict = fluid.layers.fc(input=image, size=10, act='softmax')
                return fluid.layers.cross_entropy(input=predict, label=label)
            
            def random_image_and_label_generator(height, width):
                def generator():
                    for i in range(ITER_NUM):
                        batch_image = np.random.uniform(low=0,
                                                        high=255,
                                                        size=[BATCH_SIZE, height, width])
                        batch_label = np.ones([BATCH_SIZE, 1])
                        batch_image = batch_image.astype('float32')
                        batch_label = batch_label.astype('int64')
                        yield batch_image, batch_label
                return generator
            
            image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            reader = fluid.io.PyReader(feed_list=[image, label], capacity=4, iterable=True)
            
            user_defined_generator = random_image_and_label_generator(784, 784)
            reader.decorate_batch_generator(user_defined_generator, fluid.CPUPlace())
            
            loss = network(image, label)
            executor = fluid.Executor(fluid.CPUPlace())
            executor.run(fluid.default_startup_program())
     
            for _ in range(EPOCH_NUM):
                for data in reader():
                    executor.run(feed=data, fetch_list=[loss])


next()
'''''''''

获取下一个数据。用户不应直接调用此方法。此方法用于PaddlePaddle框架内部实现Python 2.x的迭代器协议。
