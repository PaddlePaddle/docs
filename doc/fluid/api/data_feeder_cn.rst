

.. _cn_api_fluid_DataFeeder

DataFeeder
>>>>>>>>>>>>

.. py:class:: class paddle.fluid.DataFeeder(feed_list, place, program=None)

DataFeeder 将 reader 返回的数据转换为一种数据结构，该结构可以提供给 Executor 和 ParallelExecutor。DataFeeder 通常返回一个 mini batch 的 list。list 中的每个数据条目都是一个样本。每个样本是一个列表或元组，其中包含一个或多个特征。

# 
    简单用法如下:
    place = fluid.CPUPlace()
    img = fluid.layers.data(name='image', shape=[1, 28, 28])
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    feeder = fluid.DataFeeder([img, label], fluid.CPUPlace())
    result = feeder.feed([([0] * 784, [9]), ([1] * 784, [1])])

如果想在多GPU 训练模型时预先单独将数据输入 GPU，可以使用 decorate_reader 函数。

**代码示例**

.. code-block:: python

    place=fluid.CUDAPlace(0)
    feeder = fluid.DataFeeder(place=place, feed_list=[data, label])
    reader = feeder.decorate_reader(
        paddle.batch(flowers.train(), batch_size=16))

参数：
    - **feed_list**（list） - 输入模型的变量或变量名。
    - **place**（Place） - CPU 或 GPU，如果想将数据输入 GPU，请使用 fluid.CUDAPlace（i）（i 代表 GPU id），或者如果想将数据输入 CPU，请 使用 fluid.CPUPlace（）。
    - **program**（Program） - 需要使用数据的 Program，如果 Program 为 None，则使用 default_main_program（）。 默认 None。

抛出异常：
    ValueError - 如果某个变量不在该 Program 中。

**代码示例**

.. code-block:: python

    # ...
    place = fluid.CPUPlace()
    feed_list = [
        main_program.global_block().var(var_name) for var_name in feed_vars_name
    ] # feed_vars_name 是一个由变量名组成的列表。
    feeder = fluid.DataFeeder(feed_list, place)
    for data in reader():
        outs = exe.run(program=main_program,
                    feed=feeder.feed(data))

.. py:method:: feed(iterable)

根据 feed_list 和 iterable，将输入转换为 Executor 和 ParallelExecutor 所需的数据结构。

参数:	
    - **iterable** (list|tuple) – the input data.

返回:	the result of conversion.

返回类型:	dict

.. py:method:: feed_parallel(iterable, num_places=None)

使用多个 mini batch。 每个 mini batch 将提前提供给在每个设备。

参数：
    - **iterable**（list | tuple） - 输入数据。
    - **num_places**（int） - 设备数量。 默认：None。

返回：
    转换的结果。

返回类型：字典 （dict）

Notes：设备数量和小批量数量必须相同

.. py:method:: decorate_reader(reader, multi_devices, num_places=None, drop_last=True)

将输入数据转换为 reader 返回的数据，使其转换为多个 mini batch。每个设备上提供一个 mini batch


参数：
    - **reader**（fun） - 输入数据。
    - **multi_devices**（bool） - 多设备。 默认 None。
    - **num_places**（int） - 设备数量。 默认 None。
    - **drop_last**（bool） - 设备数量，默认 None。

返回：转换后的多个mini batch。

返回类型：字典（dict）

抛出异常：ValueError ，如果 drop_last 为 False 且 mini batch 不适合设备，则抛出 ValueError



