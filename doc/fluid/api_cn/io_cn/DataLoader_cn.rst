.. _cn_api_fluid_io_DataLoader:

DataLoader
-------------------------------

.. py:class:: paddle.fluid.io.DataLoader


.. py:method:: from_generator(feed_list=None, capacity=None, use_double_buffer=True, iterable=True, return_list=False)

创建一个DataLoader对象用于加载Python生成器产生的数据。数据会由Python线程预先读取，并异步送入一个队列中。

本方法创建的DataLoader对象提供了3个方法设置数据源，分别是 :code:`set_sample_generator` , :code:`set_sample_list_generator` 和
:code:`set_batch_generator` 。请查阅下述示例代码了解它们的使用方法。

如果iterable = True，本方法创建的DataLoader对象时一个Python生成器，可以for-range的方法循环迭代。

如果iterable = False，本方法创建的DataLoader对象提供 :code:`start()` 和 :code:`reset()` 方法控制数据读取过程。此模式用于兼容
``fluid.layers.py_reader`` 的使用方式。用户可使用iterable = False模式，方便地将 ``fluid.layers.py_reader`` 的代码迁移至
``fluid.io.DataLoader`` 。

参数:
    - **feed_list** (list(Variable)|tuple(Variable)) - feed变量列表，由 ``fluid.layers.data()`` 创建。
    - **capacity** (int) - DataLoader对象内部维护队列的容量大小。单位是batch数量。若reader读取速度较快，建议设置较大的capacity值。
    - **use_double_buffer** (bool) - 是否使用 ``double_buffer_reader`` 。若use_double_buffer=True，DataLoader会异步地预读取下一个batch的数据，可加速数据读取过程，但同时会占用少量的CPU/GPU存储，即一个batch输入数据的存储空间。
    - **iterable** (bool) - 所创建的DataLoader对象是否可迭代。
    - **return_list** (bool) - 每个设备上的数据是否以list形式返回。仅在iterable = True模式下有效。若return_list = False，每个设备上的返回数据均是str -> LoDTensor的映射表，其中映射表的key是每个输入变量的名称。若return_list = True，则每个设备上的返回数据均是list(LoDTensor)。推荐在静态图模式下使用return_list = False，在动态图模式下使用return_list = True。

返回: 被创建的DataLoader对象

返回类型: loader (DataLoader)

**代码示例**

.. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            BATCH_NUM = 10
            BATCH_SIZE = 16
            EPOCH_NUM = 4

            CLASS_NUM = 10

            ITERABLE = True # whether the created DataLoader object is iterable
            USE_GPU = False # whether to use GPU

            DATA_FORMAT = 'batch_generator' # data format of data source user provides

            def simple_net(image, label):
                fc_tmp = fluid.layers.fc(image, size=CLASS_NUM)
                cross_entropy = fluid.layers.softmax_with_cross_entropy(image, label)
                loss = fluid.layers.reduce_mean(cross_entropy)
                sgd = fluid.optimizer.SGD(learning_rate=1e-3)
                sgd.minimize(loss)
                return loss

            def get_random_images_and_labels(image_shape, label_shape):
                image = np.random.random(size=image_shape).astype('float32')
                label = np.random.random(size=label_shape).astype('int64')
                return image, label

            # If the data generator yields one sample each time,
            # use DataLoader.set_sample_generator to set the data source.
            def sample_generator_creator():
                def __reader__():
                    for _ in range(BATCH_NUM * BATCH_SIZE):
                        image, label = get_random_images_and_labels([784], [1])
                        yield image, label

                return __reader__

            # If the data generator yield list of samples each time,
            # use DataLoader.set_sample_list_generator to set the data source.
            def sample_list_generator_creator():
                def __reader__():
                    for _ in range(BATCH_NUM):
                        sample_list = []
                        for _ in range(BATCH_SIZE):
                            image, label = get_random_images_and_labels([784], [1])
                            sample_list.append([image, label])

                        yield sample_list

                return __reader__

            # If the data generator yields a batch each time,
            # use DataLoader.set_batch_generator to set the data source.
            def batch_generator_creator():
                def __reader__():
                    for _ in range(BATCH_NUM):
                        batch_image, batch_label = get_random_images_and_labels([BATCH_SIZE, 784], [BATCH_SIZE, 1])
                        yield batch_image, batch_label

                return __reader__

            # If DataLoader is iterable, use for loop to train the network
            def train_iterable(exe, prog, loss, loader):
                for _ in range(EPOCH_NUM):
                    for data in loader():
                        exe.run(prog, feed=data, fetch_list=[loss])

            # If DataLoader is not iterable, use start() and reset() method to control the process
            def train_non_iterable(exe, prog, loss, loader):
                for _ in range(EPOCH_NUM):
                    loader.start() # call DataLoader.start() before each epoch starts
                    try:
                        while True:
                            exe.run(prog, fetch_list=[loss])
                    except fluid.core.EOFException:
                        loader.reset() # call DataLoader.reset() after catching EOFException

            def set_data_source(loader, places):
                if DATA_FORMAT == 'sample_generator':
                    loader.set_sample_generator(sample_generator_creator(), batch_size=BATCH_SIZE, drop_last=True, places=places)
                elif DATA_FORMAT == 'sample_list_generator':
                    loader.set_sample_list_generator(sample_list_generator_creator(), places=places)
                elif DATA_FORMAT == 'batch_generator':
                    loader.set_batch_generator(batch_generator_creator(), places=places)
                else:
                    raise ValueError('Unsupported data format')

            image = fluid.layers.data(name='image', shape=[784], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')

            # Define DataLoader
            loader = fluid.io.DataLoader.from_generator(feed_list=[image, label], capacity=16, iterable=ITERABLE)

            # Define network
            loss = simple_net(image, label)

            # Set data source of DataLoader
            #
            # If DataLoader is iterable, places must be given and the number of places must be the same with device number.
            #  - If you are using GPU, call `fluid.cuda_places()` to get all GPU places.
            #  - If you are using CPU, call `fluid.cpu_places()` to get all CPU places.
            #
            # If DataLoader is not iterable, places can be None.
            places = fluid.cuda_places() if USE_GPU else fluid.cpu_places()
            set_data_source(loader, places)

            exe = fluid.Executor(places[0])
            exe.run(fluid.default_startup_program())

            prog = fluid.CompiledProgram(fluid.default_main_program()).with_data_parallel(loss_name=loss.name)

            if loader.iterable:
                train_iterable(exe, prog, loss, loader)
            else:
                train_non_iterable(exe, prog, loss, loader)


            '''
            Users can use return_list = True in dygraph mode.
            '''
            with fluid.dygraph.guard(places[0]):
                loader = fluid.io.DataLoader.from_generator(capacity=2, return_list=True)
                set_data_source(loader, places[0])
                for image, label in loader():
                    relu = fluid.layers.relu(image)
                    assert image.shape == [BATCH_SIZE, 784]
                    assert label.shape == [BATCH_SIZE, 1]
                    assert relu.shape == [BATCH_SIZE, 784]


.. py:method:: from_dataset(dataset, places, drop_last=True)

创建一个DataLoader对象用于加载Dataset产生的数据。目前，Dataset仅支持Linux系统下使用。

参数:
    - **dataset** (InMemoryDataset|QueueDataset) - Dataset对象。
    - **places** (list(CUDAPlace)|list(CPUPlace)) - DataLoader对象返回数据所在的place。
    - **drop_last** (bool) - 是否丢弃最后样本数量不足batch size的batch。若drop_last = True则丢弃，若drop_last = False则不丢弃。

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


