..  _user_guides_use_py_reader:

#############
异步数据读取
#############

除同步Feed方式外，我们提供了DataLoader。DataLoader的性能比 :ref:`user_guide_use_numpy_array_as_train_data` 更好，因为DataLoader的数据读取和模型训练过程是异步进行的，且能与 :code:`double_buffer_reader` 配合以进一步提高数据读取性能。此外， :code:`double_buffer_reader` 负责异步完成CPU Tensor到GPU Tensor的转换，一定程度上提升了数据读取效率。

创建DataLoader对象
################################

创建DataLoader对象的方式为：

.. code-block:: python

    import paddle.fluid as fluid

    image = fluid.data(name='image', dtype='float32', shape=[None, 784])
    label = fluid.data(name='label', dtype='int64', shape=[None, 1])

    ITERABLE = True

    data_loader = fluid.io.DataLoader.from_generator(
        feed_list=[image, label], capacity=64, use_double_buffer=True, iterable=ITERABLE)

其中，

- feed_list为需要输入的数据层变量列表；
- capacity为DataLoader对象的缓存区大小，单位为batch数量；
- use_double_buffer默认为True，表示使用 :code:`double_buffer_reader` 。建议开启，可提升数据读取速度；
- iterable默认为True，表示该DataLoader对象是可For-Range迭代的。推荐设置iterable=True。当iterable=True时，DataLoader与Program解耦，定义DataLoader对象不会改变Program；当iterable=False时，DataLoader会在Program中插入数据读取相关的op。

需要注意的是：`Program.clone()` (参见 :ref:`cn_api_fluid_Program` ）不能实现DataLoader对象的复制。如果您要创建多个不同DataLoader对象（例如训练和预测阶段需创建两个不同的DataLoader），则需重定义两个DataLoader对象。
若需要共享训练阶段和测试阶段的模型参数，您可以通过 :code:`fluid.unique_name.guard()` 的方式来实现。
注：Paddle采用变量名区分不同变量，且变量名是根据 :code:`unique_name` 模块中的计数器自动生成的，每生成一个变量名计数值加1。 :code:`fluid.unique_name.guard()` 的作用是重置 :code:`unique_name` 模块中的计数器，保证多次调用 :code:`fluid.unique_name.guard()` 配置网络时对应变量的变量名相同，从而实现参数共享。

下面是一个使用DataLoader配置训练阶段和测试阶段网络的例子：

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import paddle.dataset.mnist as mnist

    def network():
        image = fluid.data(name='image', dtype='float32', shape=[None, 784])
        label = fluid.data(name='label', dtype='int64', shape=[None, 1])
        loader = fluid.io.DataLoader.from_generator(feed_list=[image, label], capacity=64)

        # Definition of models
        fc = fluid.layers.fc(image, size=10)
        xe = fluid.layers.softmax_with_cross_entropy(fc, label)
        loss = fluid.layers.reduce_mean(xe)
        return loss , loader

    # Create main program and startup program for training
    train_prog = fluid.Program()
    train_startup = fluid.Program()

    with fluid.program_guard(train_prog, train_startup):
        # Use fluid.unique_name.guard() to share parameters with test network
        with fluid.unique_name.guard():
            train_loss, train_loader = network()
            adam = fluid.optimizer.Adam(learning_rate=0.01)
            adam.minimize(train_loss)

    # Create main program and startup program for testing
    test_prog = fluid.Program()
    test_startup = fluid.Program()
    with fluid.program_guard(test_prog, test_startup):
        # Use fluid.unique_name.guard() to share parameters with train network
        with fluid.unique_name.guard():
            test_loss, test_loader = network()

设置DataLoader对象的数据源
################################

DataLoader对象通过 :code:`set_sample_generator()` ， :code:`set_sample_list_generator` 和 :code:`set_batch_generator()` 方法设置其数据源。
这三个方法均接收Python生成器 :code:`generator` 作为参数，其区别在于：

- :code:`set_sample_generator()` 要求 :code:`generator` 返回的数据格式为[img_1, label_1]，其中img_1和label_1为单个样本的Numpy Array类型数据。

- :code:`set_sample_list_generator()` 要求 :code:`generator` 返回的数据格式为[(img_1, label_1), (img_2, label_2), ..., (img_n, label_n)]，其中img_i和label_i均为每个样本的Numpy Array类型数据，n为batch size。

- :code:`set_batch_generator()` 要求 :code:`generator` 返回的数据的数据格式为[batched_imgs, batched_labels]，其中batched_imgs和batched_labels为batch级的Numpy Array或LoDTensor类型数据。

值得注意的是，使用DataLoader做多GPU卡（或多CPU核）训练时，实际的总batch size为用户传入的 :code:`generator` 的batch size乘以设备数量。

当DataLoader的iterable=True（默认）时，必须给这三个方法传 :code:`places` 参数，
指定将读取的数据转换为CPU Tensor还是GPU Tensor。当DataLoader的iterable=False时，不需传places参数。

例如，假设我们有两个reader，其中fake_sample_reader每次返回一个sample的数据，fake_batch_reader每次返回一个batch的数据。

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np

    # sample级reader
    def fake_sample_reader():
        for _ in range(100):
            sample_image = np.random.random(size=(784, )).astype('float32')
            sample_label = np.random.random_integers(size=(1, ), low=0, high=9).astype('int64')
            yield sample_image, sample_label


    # batch级reader
    def fake_batch_reader():
        batch_size = 32
        for _ in range(100):
            batch_image = np.random.random(size=(batch_size, 784)).astype('float32')
            batch_label = np.random.random_integers(size=(batch_size, 1), low=0, high=9).astype('int64')
            yield batch_image, batch_label

    image1 = fluid.data(name='image1', dtype='float32', shape=[None, 784])
    label1 = fluid.data(name='label1', dtype='int64', shape=[None, 1])

    image2 = fluid.data(name='image2', dtype='float32', shape=[None, 784])
    label2 = fluid.data(name='label2', dtype='int64', shape=[None, 1])

    image3 = fluid.data(name='image3', dtype='float32', shape=[None, 784])
    label3 = fluid.data(name='label3', dtype='int64', shape=[None, 1])

对应的DataLoader设置如下：

.. code-block:: python

    import paddle
    import paddle.fluid as fluid

    ITERABLE = True
    USE_CUDA = True
    USE_DATA_PARALLEL = True

    if ITERABLE:
        # 若DataLoader可迭代，则必须设置places参数
        if USE_DATA_PARALLEL:
            # 若进行多GPU卡训练，则取所有的CUDAPlace
            # 若进行多CPU核训练，则取多个CPUPlace，本例中取了8个CPUPlace
            places = fluid.cuda_places() if USE_CUDA else fluid.cpu_places(8)
        else:
            # 若进行单GPU卡训练，则取单个CUDAPlace，本例中0代表0号GPU卡
            # 若进行单CPU核训练，则取单个CPUPlace，本例中1代表1个CPUPlace
            places = fluid.cuda_places(0) if USE_CUDA else fluid.cpu_places(1)
    else:
        # 若DataLoader不可迭代，则不需要设置places参数
        places = None

    # 使用sample级的reader作为DataLoader的数据源
    data_loader1 = fluid.io.DataLoader.from_generator(feed_list=[image1, label1], capacity=10, iterable=ITERABLE)
    data_loader1.set_sample_generator(fake_sample_reader, batch_size=32, places=places)

    # 使用sample级的reader + fluid.io.batch设置DataLoader的数据源
    data_loader2 = fluid.io.DataLoader.from_generator(feed_list=[image2, label2], capacity=10, iterable=ITERABLE)
    sample_list_reader = fluid.io.batch(fake_sample_reader, batch_size=32)
    sample_list_reader = fluid.io.shuffle(sample_list_reader, buf_size=64) # 还可以进行适当的shuffle
    data_loader2.set_sample_list_generator(sample_list_reader, places=places)

    # 使用batch级的reader作为DataLoader的数据源
    data_loader3 = fluid.io.DataLoader.from_generator(feed_list=[image3, label3], capacity=10, iterable=ITERABLE)
    data_loader3.set_batch_generator(fake_batch_reader, places=places)

使用DataLoader进行模型训练和测试
################################

使用DataLoader进行模型训练和测试的例程如下。

- 第一步，我们需组建训练网络和预测网络，并定义相应的DataLoader对象，设置好DataLoader对象的数据源。

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import paddle.dataset.mnist as mnist
    import six

    ITERABLE = True

    def network():
        # 创建数据层对象
        image = fluid.data(name='image', dtype='float32', shape=[None, 784])
        label = fluid.data(name='label', dtype='int64', shape=[None, 1])

        # 创建DataLoader对象
        reader = fluid.io.DataLoader.from_generator(feed_list=[image, label], capacity=64, iterable=ITERABLE)

        # Definition of models
        fc = fluid.layers.fc(image, size=10)
        xe = fluid.layers.softmax_with_cross_entropy(fc, label)
        loss = fluid.layers.reduce_mean(xe)
        return loss , reader

    # 创建训练的main_program和startup_program
    train_prog = fluid.Program()
    train_startup = fluid.Program()

    # 定义训练网络
    with fluid.program_guard(train_prog, train_startup):
        # fluid.unique_name.guard() to share parameters with test network
        with fluid.unique_name.guard():
            train_loss, train_loader = network()
            adam = fluid.optimizer.Adam(learning_rate=0.01)
            adam.minimize(train_loss)

    # 创建预测的main_program和startup_program
    test_prog = fluid.Program()
    test_startup = fluid.Program()

    # 定义预测网络
    with fluid.program_guard(test_prog, test_startup):
        # Use fluid.unique_name.guard() to share parameters with train network
        with fluid.unique_name.guard():
            test_loss, test_loader = network()

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    # 运行startup_program进行初始化
    exe.run(train_startup)
    exe.run(test_startup)

    # Compile programs
    train_prog = fluid.CompiledProgram(train_prog).with_data_parallel(loss_name=train_loss.name)
    test_prog = fluid.CompiledProgram(test_prog).with_data_parallel(share_vars_from=train_prog)

    # 设置DataLoader的数据源
    places = fluid.cuda_places() if ITERABLE else None

    train_loader.set_sample_list_generator(
        fluid.io.shuffle(fluid.io.batch(mnist.train(), 512), buf_size=1024), places=places)

    test_loader.set_sample_list_generator(fluid.io.batch(mnist.test(), 512), places=places)

- 第二步：根据DataLoader对象是否iterable，选用不同的方式运行网络。

若iterable=True，则DataLoader对象是一个Python的生成器，可直接for-range迭代。for-range返回的结果通过exe.run的feed参数传入执行器。

.. code-block:: python

    def run_iterable(program, exe, loss, data_loader):
        for data in data_loader():
            loss_value = exe.run(program=program, feed=data, fetch_list=[loss])
            print('loss is {}'.format(loss_value))

    for epoch_id in six.moves.range(10):
        run_iterable(train_prog, exe, train_loss, train_loader)
        run_iterable(test_prog, exe, test_loss, test_loader)

若iterable=False，则需在每个epoch开始前，调用 :code:`start()` 方法启动DataLoader对象；并在每个epoch结束时，exe.run会抛出 :code:`fluid.core.EOFException` 异常，在捕获异常后调用 :code:`reset()` 方法重置DataLoader对象的状态，
以便启动下一轮的epoch。iterable=False时无需给exe.run传入feed参数。具体方式为：

.. code-block:: python

    def run_non_iterable(program, exe, loss, data_loader):
        data_loader.start()
        try:
            while True:
                loss_value = exe.run(program=program, fetch_list=[loss])
                print('loss is {}'.format(loss_value))
        except fluid.core.EOFException:
            print('End of epoch')
            data_loader.reset()

    for epoch_id in six.moves.range(10):
        run_non_iterable(train_prog, exe, train_loss, train_loader)
        run_non_iterable(test_prog, exe, test_loss, test_loader)

