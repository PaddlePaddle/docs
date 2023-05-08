.. _cn_api_fluid_DataFeeder:

DataFeeder
-------------------------------


.. py:class:: paddle.fluid.DataFeeder(feed_list, place, program=None)






``DataFeeder`` 负责将 reader(读取器)返回的数据转成一种特殊的数据结构，使它们可以输入到 ``Executor``中。
reader 通常返回一个 minibatch 条目列表。在列表中每一条目都是一个样本（sample），它是由具有一至多个特征的列表或元组组成的。


以下是简单用法：

.. code-block:: python

  import paddle.fluid as fluid
  place = fluid.CPUPlace()
  img = fluid.layers.data(name='image', shape=[1, 28, 28])
  label = fluid.layers.data(name='label', shape=[1], dtype='int64')
  feeder = fluid.DataFeeder([img, label], fluid.CPUPlace())
  result = feeder.feed([([0] * 784, [9]), ([1] * 784, [1])])


参数
::::::::::::

    - **feed_list** (list) – 向模型输入的变量表或者变量表名
    - **place** (Place) – place 表明是向 GPU 还是 CPU 中输入数据。如果想向 GPU 中输入数据，请使用 ``fluid.CUDAPlace(i)`` (i 代表 the GPU id)；如果向 CPU 中输入数据，请使用  ``fluid.CPUPlace()``
    - **program** (Program) – 需要向其中输入数据的 Program。如果为 None，会默认使用 ``default_main_program()``。缺省值为 None


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


根据 feed_list（数据输入表）和 iterable（可遍历的数据）提供的信息，将输入数据转成一种特殊的数据结构，使它们可以输入到 ``Executor`` 和 ``ParallelExecutor`` 中。

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
