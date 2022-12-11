.. _cn_api_fluid_DataFeeder:

DataFeeder
-------------------------------


.. py:class:: paddle.fluid.DataFeeder(feed_list, place, program=None)






``DataFeeder`` 负责将reader(读取器)返回的数据转成一种特殊的数据结构，使它们可以输入到 ``Executor`` 和 ``ParallelExecutor`` 中。
reader通常返回一个minibatch条目列表。在列表中每一条目都是一个样本（sample），它是由具有一至多个特征的列表或元组组成的。


以下是简单用法：

.. code-block:: python

  import paddle.fluid as fluid
  place = fluid.CPUPlace()
  img = fluid.layers.data(name='image', shape=[1, 28, 28])
  label = fluid.layers.data(name='label', shape=[1], dtype='int64')
  feeder = fluid.DataFeeder([img, label], fluid.CPUPlace())
  result = feeder.feed([([0] * 784, [9]), ([1] * 784, [1])])

在多GPU模型训练时，如果需要提前分别向各GPU输入数据，可以使用 ``decorate_reader`` 函数。

.. code-block:: python

  import paddle
  import paddle.fluid as fluid

  place=fluid.CUDAPlace(0)
  data = fluid.layers.data(name='data', shape=[3, 224, 224], dtype='float32')
  label = fluid.layers.data(name='label', shape=[1], dtype='int64')

  feeder = fluid.DataFeeder(place=place, feed_list=[data, label])
  reader = feeder.decorate_reader(
        paddle.batch(paddle.dataset.flowers.train(), batch_size=16), multi_devices=False)



参数
::::::::::::

    - **feed_list** (list) – 向模型输入的变量表或者变量表名
    - **place** (Place) – place表明是向GPU还是CPU中输入数据。如果想向GPU中输入数据，请使用 ``fluid.CUDAPlace(i)`` (i 代表 the GPU id)；如果向CPU中输入数据，请使用  ``fluid.CPUPlace()``
    - **program** (Program) – 需要向其中输入数据的Program。如果为None，会默认使用 ``default_main_program()``。缺省值为None


抛出异常
::::::::::::

  - ``ValueError``  – 如果一些变量不在此 Program 中


代码示例
::::::::::::

.. code-block:: python

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


方法
::::::::::::
feed(iterable)
'''''''''


根据feed_list（数据输入表）和iterable（可遍历的数据）提供的信息，将输入数据转成一种特殊的数据结构，使它们可以输入到 ``Executor`` 和 ``ParallelExecutor`` 中。

**参数**

  - **iterable** (list|tuple) – 要输入的数据

**返回**
  转换结果

**返回类型**
 dict

**代码示例**

.. code-block:: python

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


feed_parallel(iterable, num_places=None)
'''''''''


该方法获取的多个minibatch，并把每个minibatch提前输入进各个设备中。

**参数**

    - **iterable** (list|tuple) – 要输入的数据
    - **num_places** (int) – 设备数目。默认为None。

**返回**
 转换结果

**返回类型**
 dict

.. note::
     设备（CPU或GPU）的数目必须等于minibatch的数目

**代码示例**

.. code-block:: python

    import numpy.random as random
    import paddle.fluid as fluid
     
    def reader(limit=10):
        for i in range(limit):
            yield [random.random([784]).astype('float32'), random.random([1]).astype('float32')],
     
    x = fluid.layers.data(name='x', shape=[1, 28, 28])
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')

    fluid.layers.elementwise_add(x, y)
     
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

decorate_reader(reader, multi_devices, num_places=None, drop_last=True)
'''''''''



将reader返回的输入数据batch转换为多个mini-batch，之后每个mini-batch都会被输入进各个设备（CPU或GPU）中。

**参数**

        - **reader** (fun) – 该参数是一个可以生成数据的函数
        - **multi_devices** (bool) – bool型，指明是否使用多个设备
        - **num_places** (int) – 如果 ``multi_devices`` 为 ``True``，可以使用此参数来设置GPU数目。如果 ``multi_devices`` 为 ``None``，该函数默认使用当前训练机所有GPU设备。默认为None。
        - **drop_last** (bool) – 如果最后一个batch的大小比 ``batch_size`` 要小，则可使用该参数来指明是否选择丢弃最后一个batch数据。默认为 ``True``

**返回**
转换结果

**返回类型**
 dict

**抛出异常**
 ``ValueError`` – 如果 ``drop_last`` 值为False并且data batch与设备不匹配时，产生此异常

**代码示例**

.. code-block:: python

    import numpy.random as random
    import paddle
    import paddle.fluid as fluid
     
    def reader(limit=5):
        for i in range(limit):
            yield (random.random([784]).astype('float32'), random.random([1]).astype('int64')),
     
    place=fluid.CPUPlace()
    data = fluid.layers.data(name='data', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
     
    feeder = fluid.DataFeeder(place=place, feed_list=[data, label])
    reader = feeder.decorate_reader(reader, multi_devices=False)
     
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    for data in reader():
        exe.run(feed=data)






