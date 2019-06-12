..  _user_guides_use_py_reader:

#############
异步数据读取
#############

除同步Feed方式外，我们提供了PyReader。PyReader的性能比 :ref:`user_guide_use_numpy_array_as_train_data` 更好，因为PyReader的数据读取和模型训练过程是异步进行的，且能与 :code:`double_buffer_reader` 配合以进一步提高数据读取性能。此外， :code:`double_buffer_reader` 负责异步完成CPU Tensor到GPU Tensor的转换，一定程度上提升了数据读取效率。

创建PyReader对象
################################

创建PyReader对象的方式为：

.. code-block:: python

    import paddle.fluid as fluid

    py_reader = fluid.layers.py_reader(capacity=64,
                                       shapes=[(-1,784), (-1,1)],
                                       dtypes=['float32', 'int64'],
                                       name='py_reader',
                                       use_double_buffer=True)

其中，capacity为PyReader对象的缓存区大小；shapes为batch各参量（如图像分类任务中的image和label）的尺寸；dtypes为batch各参量的数据类型；name为PyReader对象的名称；use_double_buffer默认为True，表示使用 :code:`double_buffer_reader` ，建议开启，可提升数据读取速度。

需要注意的是：如果您要创建多个不同PyReader对象（例如训练和预测阶段需创建两个不同的PyReader），则需要必须给不同的PyReader对象指定不同的name。这是因为PaddlePaddle采用不同的变量名区分不同的变量，而且 `Program.clone()` (参见 :ref:`cn_api_fluid_Program_clone` ）不能实现PyReader对象的复制。

.. code-block:: python

    import paddle.fluid as fluid

    train_py_reader = fluid.layers.py_reader(capacity=64,
                                             shapes=[(-1,784), (-1,1)],
                                             dtypes=['float32', 'int64'],
                                             name='train',
                                             use_double_buffer=True)

    test_py_reader = fluid.layers.py_reader(capacity=64,
                                            shapes=[(-1,3,224,224), (-1,1)],
                                            dtypes=['float32', 'int64'],
                                            name='test',
                                            use_double_buffer=True)

在使用PyReader时，如果需要共享训练阶段和测试阶段的模型参数，您可以通过 :code:`fluid.unique_name.guard()` 的方式来实现。
注：Paddle采用变量名区分不同变量，且变量名是根据 :code:`unique_name` 模块中的计数器自动生成的，每生成一个变量名计数值加1。 :code:`fluid.unique_name.guard()` 的作用是重置 :code:`unique_name` 模块中的计数器，保证多次调用 :code:`fluid.unique_name.guard()` 配置网络时对应变量的变量名相同，从而实现参数共享。

下面是一个使用PyReader配置训练阶段和测试阶段网络的例子：

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import paddle.dataset.mnist as mnist

    def network(is_train):
        # Create py_reader object and give different names
        # when is_train = True and is_train = False
        reader = fluid.layers.py_reader(
            capacity=10,
            shapes=((-1, 784), (-1, 1)),
            dtypes=('float32', 'int64'),
            name="train_reader" if is_train else "test_reader",
            use_double_buffer=True)

        # Use read_file() method to read out the data from py_reader
        img, label = fluid.layers.read_file(reader)
        ...
        # Here, we omitted the definition of loss of the model
        return loss , reader

    # Create main program and startup program for training
    train_prog = fluid.Program()
    train_startup = fluid.Program()

    with fluid.program_guard(train_prog, train_startup):
        # Use fluid.unique_name.guard() to share parameters with test network
        with fluid.unique_name.guard():
            train_loss, train_reader = network(True)
            adam = fluid.optimizer.Adam(learning_rate=0.01)
            adam.minimize(train_loss)

    # Create main program and startup program for testing
    test_prog = fluid.Program()
    test_startup = fluid.Program()
    with fluid.program_guard(test_prog, test_startup):
        # Use fluid.unique_name.guard() to share parameters with train network
        with fluid.unique_name.guard():
            test_loss, test_reader = network(False)

设置PyReader对象的数据源
################################

PyReader对象通过 :code:`decorate_paddle_reader()` 或 :code:`decorate_tensor_provider()` 方法设置其数据源。 :code:`decorate_paddle_reader()` 和 :code:`decorate_tensor_provider()` 均接收Python生成器 :code:`generator` 作为参数， :code:`generator` 内部每次通过yield的方式生成一个batch的数据。

:code:`decorate_paddle_reader()` 和 :code:`decorate_tensor_provider()` 方法的区别在于：

- :code:`decorate_paddle_reader()` 要求 :code:`generator` 返回的数据格式为[(img_1, label_1), (img_2, label_2), ..., (img_n, label_n)]，其中img_i和label_i均为每个样本的Numpy Array类型数据，n为batch size。而 :code:`decorate_tensor_provider()` 要求 :code:`generator` 返回的数据的数据格式为[batched_imgs, batched_labels]，其中batched_imgs和batched_labels为batch级的Numpy Array或LoDTensor类型数据。

- :code:`decorate_tensor_provider()` 要求 :code:`generator` 返回的数据类型、尺寸必须与配置py_reader时指定的dtypes、shapes参数相同，而 :code:`decorate_paddle_reader()` 不要求数据类型和尺寸的严格一致，其内部会完成数据类型和尺寸的转换。

具体方式为：

.. code-block:: python

    import paddle.batch
    import paddle.fluid as fluid
    import numpy as np

    BATCH_SIZE = 32

    # Case 1: Use decorate_paddle_reader() method to set the data source of py_reader
    # The generator yields Numpy-typed batched data
    def fake_random_numpy_reader():
        image = np.random.random(size=(784, ))
        label = np.random.random_integers(size=(1, ), low=0, high=9)
        yield image, label

    py_reader1 = fluid.layers.py_reader(
        capacity=10,
        shapes=((-1, 784), (-1, 1)),
        dtypes=('float32', 'int64'),
        name='py_reader1',
        use_double_buffer=True)

    py_reader1.decorate_paddle_reader(paddle.batch(fake_random_numpy_reader, batch_size=BATCH_SIZE))


    # Case 2: Use decorate_tensor_provider() method to set the data source of py_reader
    # The generator yields Tensor-typed batched data
    def fake_random_tensor_provider():
        image = np.random.random(size=(BATCH_SIZE, 784)).astype('float32')
        label = np.random.random_integers(size=(BATCH_SIZE, 1), low=0, high=9).astype('int64')
        yield image_tensor, label_tensor

    py_reader2 = fluid.layers.py_reader(
        capacity=10,
        shapes=((-1, 784), (-1, 1)),
        dtypes=('float32', 'int64'),
        name='py_reader2',
        use_double_buffer=True)

    py_reader2.decorate_tensor_provider(fake_random_tensor_provider)

使用PyReader进行模型训练和测试
################################

使用PyReader进行模型训练和测试的例程如下：

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import paddle.dataset.mnist as mnist
    import six

    def network(is_train):
        # Create py_reader object and give different names
        # when is_train = True and is_train = False
        reader = fluid.layers.py_reader(
            capacity=10,
            shapes=((-1, 784), (-1, 1)),
            dtypes=('float32', 'int64'),
            name="train_reader" if is_train else "test_reader",
            use_double_buffer=True)
        img, label = fluid.layers.read_file(reader)
        ...
        # Here, we omitted the definition of loss of the model
        return loss , reader

    # Create main program and startup program for training
    train_prog = fluid.Program()
    train_startup = fluid.Program()

    # Define train network
    with fluid.program_guard(train_prog, train_startup):
        # Use fluid.unique_name.guard() to share parameters with test network
        with fluid.unique_name.guard():
            train_loss, train_reader = network(True)
            adam = fluid.optimizer.Adam(learning_rate=0.01)
            adam.minimize(train_loss)

    # Create main program and startup program for testing
    test_prog = fluid.Program()
    test_startup = fluid.Program()

    # Define test network
    with fluid.program_guard(test_prog, test_startup):
        # Use fluid.unique_name.guard() to share parameters with train network
        with fluid.unique_name.guard():
            test_loss, test_reader = network(False)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    # Run startup program
    exe.run(train_startup)
    exe.run(test_startup)

    # Compile programs
    train_prog = fluid.CompiledProgram(train_prog).with_data_parallel(loss_name=train_loss.name)
    test_prog = fluid.CompiledProgram(test_prog).with_data_parallel(share_vars_from=train_prog)

    # Set the data source of py_reader using decorate_paddle_reader() method
    train_reader.decorate_paddle_reader(
        paddle.reader.shuffle(paddle.batch(mnist.train(), 512), buf_size=8192))

    test_reader.decorate_paddle_reader(paddle.batch(mnist.test(), 512))

    for epoch_id in six.moves.range(10):
        train_reader.start()
        try:
            while True:
                loss = exe.run(program=train_prog, fetch_list=[train_loss])
                print 'train_loss', loss
        except fluid.core.EOFException:
            print 'End of epoch', epoch_id
            train_reader.reset()

        test_reader.start()
        try:
            while True:
                loss = exe.run(program=test_prog, fetch_list=[test_loss])
                print 'test loss', loss
        except fluid.core.EOFException:
            print 'End of testing'
            test_reader.reset()

具体步骤为：

1. 在每个epoch开始前，调用 :code:`start()` 方法启动PyReader对象；

2. 在每个epoch结束时， :code:`read_file` 抛出 :code:`fluid.core.EOFException` 异常，在捕获异常后调用 :code:`reset()` 方法重置PyReader对象的状态，以便启动下一轮的epoch。
