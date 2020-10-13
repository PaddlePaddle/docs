.. _cn_api_fluid_io_DataLoader:

DataLoader
-------------------------------

.. py:class:: paddle.fluid.io.DataLoader(dataset, feed_list=None, places=None, return_list=False, batch_sampler=None, batch_size=1, shuffle=False, drop_last=False, collate_fn=None, num_workers=0, use_buffer_reader=True, use_shared_memory=False, timeout=0, worker_init_fn=None)

DataLoader返回一个迭代器，该迭代器根据 ``batch_sampler`` 给定的顺序迭代一次给定的 ``dataset``

DataLoader支持单进程和多进程的数据加载方式，当 ``num_workers`` 大于0时，将使用多进程方式异步加载数据。

DataLoader当前仅支持 ``map-style`` 的数据集(可通过下标索引样本)， ``map-style`` 的数据集请参考 ``paddle.io.Dataset`` 。

``batch_sampler`` 请参考 ``fluid.io.BatchSampler``

参数:
    - **dataset** (Dataset) - DataLoader从此参数给定数据集中加载数据，此参数必须是 ``paddle.io.Dataset`` 或 ``paddle.io.IterableDataset`` 的一个子类实例。
    - **feed_list** (list(Tensor)|tuple(Tensor)) - feed变量列表，由 ``fluid.layers.data()`` 创建。当 ``return_list`` 为False时，此参数必须设置。默认值为None。
    - **places** (list(Place)|tuple(Place)) - 数据需要放置到的Place列表。在静态图和动态图模式中，此参数均必须设置。在动态图模式中，此参数列表长度必须是1。默认值为None。
    - **return_list** (bool) - 每个设备上的数据是否以list形式返回。若return_list = False，每个设备上的返回数据均是str -> Tensor的映射表，其中映射表的key是每个输入变量的名称。若return_list = True，则每个设备上的返回数据均是list(Tensor)。在动态图模式下，此参数必须为True。默认值为False。
    - **batch_sampler** (BatchSampler) - ``fluid.io.BatchSampler`` 或其子类的实例，DataLoader通过 ``batch_sampler`` 产生的mini-batch索引列表来 ``dataset`` 中索引样本并组成mini-batch。默认值为None。
    - **batch_size** (int) - 每mini-batch中样本个数，为 ``batch_sampler`` 的替代参数，若 ``batch_sampler`` 未设置，会根据 ``batch_size`` ``shuffle`` ``drop_last`` 创建一个 ``fluid.io.BatchSampler`` 。默认值为1。
    - **shuffle** (bool) - 生成mini-batch索引列表时是否对索引打乱顺序，为 ``batch_sampler`` 的替代参数，若 ``batch_sampler`` 未设置，会根据 ``batch_size`` ``shuffle`` ``drop_last`` 创建一个 ``fluid.io.BatchSampler`` 。默认值为False。
    - **drop_last** (bool) - 是否丢弃因数据集样本数不能被 ``batch_size`` 整除而产生的最后一个不完整的mini-batch，为 ``batch_sampler`` 的替代参数，若 ``batch_sampler`` 未设置，会根据 ``batch_size`` ``shuffle`` ``drop_last`` 创建一个 ``fluid.io.BatchSampler`` 。默认值为False。
    - **collate_fn** (callable) - 通过此参数指定如果将样本列表组合为mini-batch数据，当 ``collate_fn`` 为None时，默认为将样本个字段在第0维上堆叠(同 ``np.stack(..., axis=0)`` )为mini-batch的数据。默认值为None。
    - **num_workers** (int) - 用于加载数据的子进程个数，若为0即为不开启子进程，在主进程中进行数据加载。默认值为0。
    - **use_buffer_reader** (bool) - 是否使用缓存读取器 。若 ``use_buffer_reader`` 为True，DataLoader会异步地预读取下一个mini-batch的数据，可加速数据读取过程，但同时会占用少量的CPU/GPU存储，即一个batch输入数据的存储空间。默认值为True。
    - **use_shared_memory** (bool) - 是否使用共享内存来提升子进程将数据放入进程间队列的速度，该参数尽在多进程模式下有效(即 ``num_workers > 0`` )，请确认机器上有足够的共享内存空间(如Linux系统下 ``/dev/shm/`` 目录空间大小)再设置此参数。默认为False。
    - **timeout** (int) - 从子进程输出队列获取mini-batch数据的超时时间。默认值为0。
    - **worker_init_fn** (callable) - 子进程初始化函数，此函数会被子进程初始化时被调用，并传递 ``worker id`` 作为参数。默认值为None。

返回：迭代 ``dataset`` 数据的迭代器，迭代器返回的数据中的每个元素都是一个Tensor。

返回类型: DataLoader

**代码示例**

.. code-block:: python


    import numpy as np

    import paddle
    import paddle.fluid as fluid
    from paddle.io import Dataset, BatchSampler, DataLoader

    BATCH_NUM = 20
    BATCH_SIZE = 16
    EPOCH_NUM = 4

    IMAGE_SIZE = 784
    CLASS_NUM = 10

    USE_GPU = False # whether use GPU to run model

    # define a random dataset
    class RandomDataset(Dataset):
        def __init__(self, num_samples):
            self.num_samples = num_samples

        def __getitem__(self, idx):
            image = np.random.random([IMAGE_SIZE]).astype('float32')
            label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
            return image, label

        def __len__(self):
            return self.num_samples

    dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)

    # get places
    places = fluid.cuda_places() if USE_GPU else fluid.cpu_places()

    # --------------------- dygraph mode --------------------

    class SimpleNet(fluid.dygraph.Layer):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc = fluid.dygraph.nn.Linear(IMAGE_SIZE, CLASS_NUM, act='softmax')

        def forward(self, image, label=None):
            return self.fc(image)

    with fluid.dygraph.guard(places[0]):
        simple_net = SimpleNet()
        opt = fluid.optimizer.SGD(learning_rate=1e-3,
                                  parameter_list=simple_net.parameters())

        loader = DataLoader(dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            drop_last=True,
                            num_workers=2)

        for e in range(EPOCH_NUM):
            for i, (image, label) in enumerate(loader()):
                out = simple_net(image)
                loss = fluid.layers.cross_entropy(out, label)
                avg_loss = fluid.layers.reduce_mean(loss)
                avg_loss.backward()
                opt.minimize(avg_loss)
                simple_net.clear_gradients()
                print("Epoch {} batch {}: loss = {}".format(e, i, np.mean(loss.numpy())))

    # -------------------------------------------------------

    # -------------------- static graph ---------------------

    paddle.enable_static()

    def simple_net(image, label):
        fc_tmp = fluid.layers.fc(image, size=CLASS_NUM, act='softmax')
        cross_entropy = fluid.layers.softmax_with_cross_entropy(image, label)
        loss = fluid.layers.reduce_mean(cross_entropy)
        sgd = fluid.optimizer.SGD(learning_rate=1e-3)
        sgd.minimize(loss)
        return loss

    image = fluid.data(name='image', shape=[None, IMAGE_SIZE], dtype='float32')
    label = fluid.data(name='label', shape=[None, 1], dtype='int64')

    loss = simple_net(image, label)

    exe = fluid.Executor(places[0])
    exe.run(fluid.default_startup_program())

    prog = fluid.CompiledProgram(fluid.default_main_program()).with_data_parallel(loss_name=loss.name)

    loader = DataLoader(dataset,
                        feed_list=[image, label],
                        batch_size=BATCH_SIZE, 
                        shuffle=True,
                        drop_last=True,
                        num_workers=2)

    for e in range(EPOCH_NUM):
        for i, data in enumerate(loader()):
            l = exe.run(prog, feed=data, fetch_list=[loss], return_numpy=True)
            print("Epoch {} batch {}: loss = {}".format(e, i, l[0][0]))

    # -------------------------------------------------------

.. py:method:: from_generator(feed_list=None, capacity=None, use_double_buffer=True, iterable=True, return_list=False, use_multiprocess=False, drop_last=True)

.. note::
    框架保证DataLoader的数据加载顺序与用户提供的数据源读取顺序一致。

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
    - **use_multiprocess** (bool) - 设置是否是用多进程加速动态图的数据载入过程。注意：该参数的设置仅在动态图模式下有效, 在静态图模式下，该参数设置与否均无任何影响。默认值为False。
    - **drop_last** (bool): 是否丢弃最后的不足CPU/GPU设备数的批次。默认值为True。在网络训练时，用户不能设置drop_last=False，此时所有CPU/GPU设备均应从DataLoader中读取到数据。在网络预测时，用户可以设置drop_last=False，此时最后不足CPU/GPU设备数的批次可以进行预测。

返回: 被创建的DataLoader对象

返回类型: loader (DataLoader)

**代码示例 1**

.. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np

            BATCH_NUM = 10
            BATCH_SIZE = 16
            EPOCH_NUM = 4

            CLASS_NUM = 10

            ITERABLE = True # whether the created DataLoader object is iterable
            USE_GPU = False # whether to use GPU

            DATA_FORMAT = 'batch_generator' # data format of data source user provides

            paddle.enable_static()

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


**代码示例 2**

.. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np
            import os

            # We use 2 CPU cores to run inference network
            os.environ['CPU_NUM'] = '2'

            paddle.enable_static()

            # The data source has only 3 batches, which can not be
            # divided evenly to each CPU core
            def batch_generator():
                for i in range(3):
                    yield np.array([i+1]).astype('float32'),

            x = fluid.data(name='x', shape=[None], dtype='float32')
            y = x * x

            def run_inference(drop_last):
                loader = fluid.io.DataLoader.from_generator(feed_list=[x],
                        capacity=8, drop_last=drop_last)
                loader.set_batch_generator(batch_generator, fluid.cpu_places())

                exe = fluid.Executor(fluid.CPUPlace())
                prog = fluid.CompiledProgram(fluid.default_main_program())
                prog = prog.with_data_parallel()

                result = []
                for data in loader():
                    each_ret, = exe.run(prog, feed=data, fetch_list=[y])
                    result.extend(each_ret)
                return result

            # Set drop_last to True, so that the last batch whose
            # number is less than CPU core number would be discarded.
            print(run_inference(drop_last=True)) # [1.0, 4.0]

            # Set drop_last to False, so that the last batch whose
            # number is less than CPU core number can be tested.
            print(run_inference(drop_last=False)) # [1.0, 4.0, 9.0]


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

            import paddle
            import paddle.fluid as fluid

            paddle.enable_static()

            image = fluid.layers.data(name='image', shape=[784], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')

            dataset = fluid.DatasetFactory().create_dataset("QueueDataset")
            dataset.set_batch_size(32)
            dataset.set_filelist(['a.txt', 'b.txt', 'c.txt'])
            dataset.set_use_var([image, label])
            dataset.set_pipe_command('cat')


