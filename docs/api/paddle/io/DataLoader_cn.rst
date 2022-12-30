.. _cn_api_fluid_io_DataLoader:

DataLoader
-------------------------------

.. py:class:: paddle.io.DataLoader(dataset, feed_list=None, places=None, return_list=False, batch_sampler=None, batch_size=1, shuffle=False, drop_last=False, collate_fn=None, num_workers=0, use_buffer_reader=True, use_shared_memory=True, prefetch_factor=2, timeout=0, worker_init_fn=None)

DataLoader 返回一个迭代器，该迭代器根据 ``batch_sampler`` 给定的顺序迭代一次给定的 ``dataset``

DataLoader 支持单进程和多进程的数据加载方式，当 ``num_workers`` 大于 0 时，将使用多进程方式异步加载数据。

DataLoader 当前支持 ``map-style`` 和 ``iterable-style`` 的数据集，``map-style`` 的数据集可通过下标索引样本，请参考 ``paddle.io.Dataset`` ； ``iterable-style`` 数据集只能迭代式地获取样本，类似 Python 迭代器，请参考 ``paddle.io.IterableDataset`` 。

.. note::

    当前还不支持在子进程中进行 GPU Tensor 的操作，请不要在子进程流程中使用 GPU Tensor，例如 ``dataset`` 中的预处理，``collate_fn`` 等，``numpy array`` 和 CPU Tensor 操作已支持。

``batch_sampler`` 请参考 ``paddle.io.BatchSampler``

**禁用自动组 batch**

在如 NLP 等任务中，用户需求自定义组 batch 的方式，不希望 ``DataLoader`` 自动组 batch， ``DataLoader`` 支持在 ``batch_size`` 和 ``batch_sampler`` 均为 None 的时候禁用自动组 batch 功能，此时需求从 ``dataset`` 中获取的数据为已经组好 batch 的数据，该数据将不做任何处理直接传到 ``collate_fn`` 或 ``default_collate_fn`` 中。

.. note::

    当禁用自动组 batch 时，``default_collate_fn`` 将不对输入数据做任何处理。

参数
::::::::::::

    - **dataset** (Dataset) - DataLoader 从此参数给定数据集中加载数据，此参数必须是 ``paddle.io.Dataset`` 或 ``paddle.io.IterableDataset`` 的一个子类实例。
    - **feed_list** (list(Tensor)|tuple(Tensor)，可选) - feed 变量列表，由 ``paddle.static.data()`` 创建。当 ``return_list`` 为 False 时，此参数必须设置。默认值为 None。
    - **places** (list(Place)|tuple(Place)，可选) - 数据需要放置到的 Place 列表。在静态图和动态图模式中，此参数均必须设置。在动态图模式中，此参数列表长度必须是 1。默认值为 None。
    - **return_list** (bool，可选) - 每个设备上的数据是否以 list 形式返回。若 return_list = False，每个设备上的返回数据均是 str -> Tensor 的映射表，其中映射表的 key 是每个输入变量的名称。若 return_list = True，则每个设备上的返回数据均是 list(Tensor)。在动态图模式下，此参数必须为 True。默认值为 False。
    - **batch_sampler** (BatchSampler，可选) - ``paddle.io.BatchSampler`` 或其子类的实例，DataLoader 通过 ``batch_sampler`` 产生的 mini-batch 索引列表来 ``dataset`` 中索引样本并组成 mini-batch。默认值为 None。
    - **batch_size** (int|None，可选) - 每 mini-batch 中样本个数，为 ``batch_sampler`` 的替代参数，若 ``batch_sampler`` 未设置，会根据 ``batch_size`` ``shuffle`` ``drop_last`` 创建一个 ``paddle.io.BatchSampler``。默认值为 1。
    - **shuffle** (bool，可选) - 生成 mini-batch 索引列表时是否对索引打乱顺序，为 ``batch_sampler`` 的替代参数，若 ``batch_sampler`` 未设置，会根据 ``batch_size`` ``shuffle`` ``drop_last`` 创建一个 ``paddle.io.BatchSampler``。默认值为 False。
    - **drop_last** (bool，可选) - 是否丢弃因数据集样本数不能被 ``batch_size`` 整除而产生的最后一个不完整的 mini-batch，为 ``batch_sampler`` 的替代参数，若 ``batch_sampler`` 未设置，会根据 ``batch_size`` ``shuffle`` ``drop_last`` 创建一个 ``paddle.io.BatchSampler``。默认值为 False。
    - **collate_fn** (callable，可选) - 通过此参数指定如果将样本列表组合为 mini-batch 数据，当 ``collate_fn`` 为 None 时，默认为将样本个字段在第 0 维上堆叠(同 ``np.stack(..., axis=0)`` )为 mini-batch 的数据。默认值为 None。
    - **num_workers** (int，可选) - 用于加载数据的子进程个数，若为 0 即为不开启子进程，在主进程中进行数据加载。默认值为 0。
    - **use_buffer_reader** (bool，可选) - 是否使用缓存读取器。若 ``use_buffer_reader`` 为 True，DataLoader 会异步地预读取一定数量（默认读取下一个）的 mini-batch 的数据，可加速数据读取过程，但同时会占用少量的 CPU/GPU 存储，即一个 batch 输入数据的存储空间。默认值为 True。
    - **prefetch_factor** (int，可选) - 缓存的 mini-batch 的个数。若 ``use_buffer_reader`` 为 True，DataLoader 会异步地预读取 ``prefetch_factor`` 个 mini-batch。默认值为 2。
    - **use_shared_memory** (bool，可选) - 是否使用共享内存来提升子进程将数据放入进程间队列的速度，该参数尽在多进程模式下有效(即 ``num_workers > 0`` )，请确认机器上有足够的共享内存空间(如 Linux 系统下 ``/dev/shm/`` 目录空间大小)再设置此参数。默认为 True。
    - **timeout** (int，可选) - 从子进程输出队列获取 mini-batch 数据的超时时间。默认值为 0。
    - **worker_init_fn** (callable，可选) - 子进程初始化函数，此函数会被子进程初始化时被调用，并传递 ``worker id`` 作为参数。默认值为 None。

返回
::::::::::::
DataLoader，迭代 ``dataset`` 数据的迭代器，迭代器返回的数据中的每个元素都是一个 Tensor。


代码示例
::::::::::::

.. code-block:: python

    import numpy as np

    import paddle
    import paddle.nn as nn
    import paddle.nn.functional as F
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

    class SimpleNet(nn.Layer):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(IMAGE_SIZE, CLASS_NUM)

        def forward(self, image, label=None):
            return self.fc(image)

    simple_net = SimpleNet()
    opt = paddle.optimizer.SGD(learning_rate=1e-3,
                              parameters=simple_net.parameters())

    loader = DataLoader(dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        drop_last=True,
                        num_workers=2)

    for e in range(EPOCH_NUM):
        for i, (image, label) in enumerate(loader()):
            out = simple_net(image)
            loss = F.cross_entropy(out, label)
            avg_loss = paddle.mean(loss)
            avg_loss.backward()
            opt.minimize(avg_loss)
            simple_net.clear_gradients()
            print("Epoch {} batch {}: loss = {}".format(e, i, np.mean(loss.numpy())))

方法
::::::::::::
from_generator(feed_list=None, capacity=None, use_double_buffer=True, iterable=True, return_list=False, use_multiprocess=False, drop_last=True)
'''''''''

.. warning::
    这个 API 将在未来版本废弃，推荐使用支持多进程并发加速的 ``paddle.io.DataLoader``

.. note::
    框架保证 DataLoader 的数据加载顺序与用户提供的数据源读取顺序一致。

创建一个 DataLoader 对象用于加载 Python 生成器产生的数据。数据会由 Python 线程预先读取，并异步送入一个队列中。

本方法创建的 DataLoader 对象提供了 3 个方法设置数据源，分别是 :code:`set_sample_generator` , :code:`set_sample_list_generator` 和
:code:`set_batch_generator`。请查阅下述示例代码了解它们的使用方法。

如果 iterable = True，本方法创建的 DataLoader 对象是一个 Python 生成器，可以 for-range 的方法循环迭代。

如果 iterable = False，本方法创建的 DataLoader 对象提供 :code:`start()` 和 :code:`reset()` 方法控制数据读取过程。

**参数**

    - **feed_list** (list(Tensor)|tuple(Tensor)) - feed 变量列表，由 ``paddle.static.data()`` 创建。
    - **capacity** (int) - DataLoader 对象内部维护队列的容量大小。单位是 batch 数量。若 reader 读取速度较快，建议设置较大的 capacity 值。
    - **use_double_buffer** (bool，可选) - 是否使用 ``double_buffer_reader``。若 use_double_buffer=True，DataLoader 会异步地预读取下一个 batch 的数据，可加速数据读取过程，但同时会占用少量的 CPU/GPU 存储，即一个 batch 输入数据的存储空间。
    - **iterable** (bool，可选) - 所创建的 DataLoader 对象是否可迭代。
    - **return_list** (bool，可选) - 每个设备上的数据是否以 list 形式返回。仅在 iterable = True 模式下有效。若 return_list = False，每个设备上的返回数据均是 str -> Tensor 的映射表，其中映射表的 key 是每个输入变量的名称。若 return_list = True，则每个设备上的返回数据均是 list(Tensor)。推荐在静态图模式下使用 return_list = False，在动态图模式下使用 return_list = True。
    - **use_multiprocess** (bool，可选) - 设置是否是用多进程加速动态图的数据载入过程。注意：该参数的设置仅在动态图模式下有效，在静态图模式下，该参数设置与否均无任何影响。默认值为 False。
    - **drop_last** (bool，可选)：是否丢弃最后的不足 CPU/GPU 设备数的批次。默认值为 True。在网络训练时，用户不能设置 drop_last=False，此时所有 CPU/GPU 设备均应从 DataLoader 中读取到数据。在网络预测时，用户可以设置 drop_last=False，此时最后不足 CPU/GPU 设备数的批次可以进行预测。

**返回**

 被创建的 DataLoader 对象。


**代码示例 1**

.. code-block:: python

    '''
    Example in static graph mode
    '''
    import numpy as np

    import paddle
    import paddle.static as static
    import paddle.nn.functional as F


    BATCH_NUM = 10
    BATCH_SIZE = 16
    EPOCH_NUM = 4

    CLASS_NUM = 10

    ITERABLE = True # whether the created DataLoader object is iterable
    USE_GPU = False # whether to use GPU

    DATA_FORMAT = 'batch_generator' # data format of data source user provides

    paddle.enable_static()

    def simple_net(image, label):
        fc_tmp = static.nn.fc(image, size=CLASS_NUM)
        cross_entropy = F.softmax_with_cross_entropy(image, label)
        loss = paddle.mean(cross_entropy)
        sgd = paddle.optimizer.SGD(learning_rate=1e-3)
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
            except paddle.core.EOFException:
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

    image = static.data(name='image', shape=[None, 784], dtype='float32')
    label = static.data(name='label', shape=[None, 1], dtype='int64')

    # Define DataLoader
    loader = paddle.io.DataLoader.from_generator(feed_list=[image, label], capacity=16, iterable=ITERABLE)

    # Define network
    loss = simple_net(image, label)

    # Set data source of DataLoader
    #
    # If DataLoader is iterable, places must be given and the number of places must be the same with device number.
    #  - If you are using GPU, call `paddle.static.cuda_places()` to get all GPU places.
    #  - If you are using CPU, call `paddle.static.cpu_places()` to get all CPU places.
    #
    # If DataLoader is not iterable, places can be None.
    places = static.cuda_places() if USE_GPU else static.cpu_places()
    set_data_source(loader, places)

    exe = static.Executor(places[0])
    exe.run(static.default_startup_program())

    prog = static.CompiledProgram(static.default_main_program()).with_data_parallel(loss_name=loss.name)

    if loader.iterable:
        train_iterable(exe, prog, loss, loader)
    else:
        train_non_iterable(exe, prog, loss, loader)


**代码示例 2**

.. code-block:: python

    '''
    Example in dynamic graph mode.
    '''
    import numpy as np

    import paddle
    import paddle.nn as nn
    import paddle.optimizer as opt
    import paddle.distributed as dist

    BATCH_SIZE = 16
    BATCH_NUM = 4
    EPOCH_NUM = 4

    IMAGE_SIZE = 784
    CLASS_NUM = 1

    USE_GPU = False # whether to use GPU

    def _get_random_images_and_labels(image_shape, label_shape):
            image = np.random.random(size=image_shape).astype('float32')
            label = np.random.random(size=label_shape).astype('int64')
            return image, label

    def __reader__():
            for _ in range(BATCH_NUM):
                batch_image, batch_label = _get_random_images_and_labels(
                    [BATCH_SIZE, IMAGE_SIZE], [BATCH_SIZE, CLASS_NUM])
                yield batch_image, batch_label

    def random_batch_reader():
        return __reader__

    class LinearNet(nn.Layer):
        def __init__(self):
            super().__init__()
            self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

        @paddle.jit.to_static
        def forward(self, x):
            return self._linear(x)

    # set device
    paddle.set_device('gpu' if USE_GPU else 'cpu')

    # create network
    layer = LinearNet()
    dp_layer = paddle.DataParallel(layer)
    loss_fn = nn.CrossEntropyLoss()
    adam = opt.Adam(learning_rate=0.001, parameters=dp_layer.parameters())

    # create data loader
    loader = paddle.io.DataLoader.from_generator(capacity=5)
    loader.set_batch_generator(random_batch_reader())

    for epoch_id in range(EPOCH_NUM):
        for batch_id, (image, label) in enumerate(loader()):
            out = layer(image)
            loss = loss_fn(out, label)

            loss.backward()

            adam.step()
            adam.clear_grad()
            print("Epoch {} batch {}: loss = {}".format(
                epoch_id, batch_id, np.mean(loss.numpy())))

**代码示例 3**

.. code-block:: python

    '''
    Example of `drop_last` using in static graph multi-cards mode
    '''
    import paddle
    import paddle.static as static
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

    x = static.data(name='x', shape=[None], dtype='float32')
    y = x * x

    def run_inference(drop_last):
        loader = paddle.io.DataLoader.from_generator(feed_list=[x],
                capacity=8, drop_last=drop_last)
        loader.set_batch_generator(batch_generator, static.cpu_places())

        exe = static.Executor(paddle.CPUPlace())
        prog = static.CompiledProgram(static.default_main_program())
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


from_dataset(dataset, places, drop_last=True)
'''''''''

.. warning::
    这个 API 将在未来版本废弃，推荐使用支持多进程并发加速的 ``paddle.io.DataLoader``

创建一个 DataLoader 对象用于加载 Dataset 产生的数据。目前，Dataset 仅支持 Linux 系统下使用。

**参数**

    - **dataset** (InMemoryDataset|QueueDataset) - Dataset 对象。
    - **places** (list(CUDAPlace)|list(CPUPlace)) - DataLoader 对象返回数据所在的 place。
    - **drop_last** (bool，可选) - 是否丢弃最后样本数量不足 batch size 的 batch。若 drop_last = True 则丢弃，若 drop_last = False 则不丢弃。

**返回**

 被创建的 DataLoader 对象，可以 for-range 的方式循环迭代。


**代码示例**

.. code-block:: python

    import paddle
    import paddle.static as static

    paddle.enable_static()

    image = static.data(name='image', shape=[None, 784], dtype='float32')
    label = static.data(name='label', shape=[None, 1], dtype='int64')

    dataset = paddle.distributed.QueueDataset()
    dataset.init(
        batch_size=32,
        pipe_command='cat',
        use_var=[image, label])
    dataset.set_filelist(['a.txt', 'b.txt', 'c.txt'])

    loader = paddle.io.DataLoader.from_dataset(dataset, static.cpu_places())
