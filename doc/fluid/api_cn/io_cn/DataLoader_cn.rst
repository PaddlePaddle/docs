.. _cn_api_fluid_io_DataLoader:

DataLoader
-------------------------------

.. py:class:: paddle.fluid.io.DataLoader


.. py:method:: from_generator(feed_list=None, capacity=None, use_double_buffer=True, iterable=True, return_list=False)

创建一个DataLoader对象用于加载Python生成器产生的数据。数据会由Python线程预先读取，并异步送入一个队列中。

本方法创建的DataLoader对象提供了3个方法设置数据源，分别是 :code:`set_sample_generator` , :code:`set_sample_list_generator` 和
:code:`set_batch_generator` ，这3个方法效果与 :code:`PyReader.decorate_sample_generator` ,
:code:`PyReader.decorate_sample_list_generator` 和 :code:`PyReader.decorate_batch_generator` 相同。

如果iterable = False，本方法创建的DataLoader对象提供 :code:`start()` 和 :code:`reset()` 方法控制数据读取过程，这些方法与不可迭代模式下的
:code:`PyReader.start()` 和 :code:`PyReader.reset()` 效果相同。

如果iterable = True，本方法创建的DataLoader对象时一个Python生成器，可以for-range的方法循环迭代。

参数:
    - **feed_list** (list(Variable)|tuple(Variable)) - feed变量列表，由 ``fluid.layers.data()`` 创建。
    - **capacity** (int) - DataLoader对象内部维护队列的容量大小。单位是batch数量。
    - **use_double_buffer** (bool) - 是否使用 ``double_buffer_reader`` 来加速数据输入。
    - **iterable** (bool) - 所创建的DataLoader对象是否可迭代。
    - **return_list** (bool) - 是否以list的形式返回值。

返回: 被创建的DataLoader对象

返回类型: loader (DataLoader)

**代码示例**

.. code-block:: python

            import paddle.fluid as fluid

            image = fluid.layers.data(name='image', shape=[784], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            loader = fluid.io.DataLoader.from_generator(feed_list=[image, label], capacity=16)


.. py:method:: from_dataset(dataset, places, drop_last=True)

创建一个DataLoader对象用于加载Dataset产生的数据。目前，Dataset仅支持Linux系统下使用。

参数:
    - **dataset** (InMemoryDataset|QueueDataset) - Dataset对象。
    - **places** (list(CUDAPlace)|list(CPUPlace)) - DataLoader对象返回数据所在的place。
    - **drop_last** (bool) - 是否丢弃最后一个样本数量不足batch size的batch。

返回: 被创建的DataLoader对象，可以for-range的方式循环迭代

返回类型: loader (DataLoader)

**代码示例**

.. code-block:: python

            import paddle.fluid as fluid

            image = fluid.layers.data(name='image', shape=[784], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')

            dataset = fluid.DatasetFactory().create_dataset("QueueDataset")
            dataset.set_batch_size(32)
            dataset.set_filelist(['a.txt', 'b.txt', 'c.txt'])
            dataset.set_use_var([image, label])
            dataset.set_pipe_command('cat')

            loader = fluid.io.DataLoader.from_dataset(dataset, fluid.cpu_places())


