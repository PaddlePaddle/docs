..  _user_guides_use_py_reader:

############################
使用PyReader读取训练和测试数据
############################

Paddle Fluid支持PyReader，实现Python端往C++端导入数据的功能。与 :ref:`user_guide_use_numpy_array_as_train_data` 不同，在使用PyReader时，Python端导入数据的过程和C++端 :code:`Executor::Run()` 读取数据的过程是异步进行的，且能与 :code:`double_buffer_reader` 配合以进一步提高数据读取性能。

创建PyReader对象
################################

用户创建PyReader对象的方式为：

.. code-block:: python

    import paddle.fluid as fluid

    py_reader = fluid.layers.py_reader(capacity=64,
                                       shapes=[(-1,3,224,224), (-1,1)],
                                       dtypes=['float32', 'int64'],
                                       name='py_reader',
                                       use_double_buffer=True)

其中，capacity为PyReader对象的缓存区大小；shapes为batch各参量（如图像分类任务中的image和label)的尺寸；dtypes为batch各参量的数据类型；name为PyReader对象的名称；use_double_buffer默认为True，表示使用 :code:`double_buffer_reader` 。

若要创建多个不同的PyReader对象（如训练阶段和测试阶段往往需创建两个不同的PyReader对象），必须给不同的PyReader对象指定不同的name。比如，在同一任务中创建训练阶段和测试阶段的PyReader对象的方式为：

.. code-block:: python

    import paddle.fluid as fluid

    train_py_reader = fluid.layers.py_reader(capacity=64,
                                             shapes=[(-1,3,224,224), (-1,1)],
                                             dtypes=['float32', 'int64'],
                                             name='train',
                                             use_double_buffer=True)

    test_py_reader = fluid.layers.py_reader(capacity=64,
                                            shapes=[(-1,3,224,224), (-1,1)],
                                            dtypes=['float32', 'int64'],
                                            name='test',
                                            use_double_buffer=True)

注意， :code:`Program.clone()` 方法不能实现PyReader对象的复制，因此必须用以上方式创建训练阶段和测试阶段的不同
PyReader对象。

由于 :code:`Program.clone()` 无法实现PyReader对象的复制，因此用户需通过 :code:`fluid.unique_name.guard()`
的方式实现训练阶段和测试阶段模型参数的共享，具体方式为：

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import paddle.dataset.mnist as mnist

    import numpy

    def network(is_train):
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

    train_prog = fluid.Program()
    train_startup = fluid.Program()

    with fluid.program_guard(train_prog, train_startup):
        with fluid.unique_name.guard():
            train_loss, train_reader = network(True)
            adam = fluid.optimizer.Adam(learning_rate=0.01)
            adam.minimize(train_loss)

    test_prog = fluid.Program()
    test_startup = fluid.Program()
    with fluid.program_guard(test_prog, test_startup):
        with fluid.unique_name.guard():
            test_loss, test_reader = network(False)

设置PyReader对象的数据源
################################
<<<<<<< Updated upstream
PyReader对象提供 :code:`decorate_tensor_provider` 和 :code:`decorate_paddle_reader` 方法，它们均接收一个Python生成器 :code:`generator` 对象作为数据源，两个方法的区别在于：

1. :code:`decorate_tensor_provider` 方法：要求 :code:`generator` 每次产生一个 :code:`list` 或 :code:`tuple` 对象， :code:`list` 或 :code:`tuple` 对象中的每个元素为 :code:`LoDTensor` 类型或Numpy数组类型，且 :code:`LoDTensor` 或Numpy数组的 :code:`shape` 必须与创建PyReader对象时指定的 :code:`shapes` 参数完全一致。

2. :code:`decorate_paddle_reader` 方法：要求 :code:`generator` 每次产生一个 :code:`list` 或 :code:`tuple` 对象， :code:`list` 或 :code:`tuple` 对象中的每个元素为Numpy数组类型，但Numpy数组的 :code:`shape` 不必与创建PyReader对象时指定的 :code:`shapes` 参数完全一致， :code:`decorate_paddle_reader` 方法内部会对其进行 :code:`reshape` 操作。
=======
PyReader对象通过 :code:`decorate_paddle_reader()` 或 :code:`decorate_tensor_provider()` 方法设置其数据源。 :code:`decorate_paddle_reader()` 和 :code:`decorate_tensor_provider()` 均接收Python生成器 :code:`generator` 作为参数， :code:`generator` 内部每次通过yield的方式生成一个batch的数据。

:code:`decorate_paddle_reader()` 和 :code:`decorate_tensor_provider()` 方法的区别在于：

- :code:`decorate_paddle_reader()` 要求 :code:`generator` 返回的数据格式为[(img_1, label_1), (img_2, label_2), ..., (img_n, label_n)]，其中img_i和label_i均为每个样本的Numpy Array类型数据，n为batch size。而 :code:`decorate_tensor_provider()` 要求 :code:`generator` 返回的数据的数据格式为[batched_imgs, batched_labels]，其中batched_imgs和batched_labels为batch级的Numpy Array或LoDTensor类型数据。

- :code:`decorate_tensor_provider()` 要求 :code:`generator` 返回的LoDTensor的数据类型、尺寸必须与配置py_reader时指定的dtypes、shapes参数相同，而 :code:`decorate_paddle_reader()` 不要求数据类型和尺寸的严格一致，其内部会完成数据类型和尺寸的转换。

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

    py_reader1.decorate_paddle_reader(paddle.batch(fake_random_reader, batch_size=BATCH_SIZE))

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
>>>>>>> Stashed changes

使用PyReader进行模型训练和测试
################################

具体方式为（接上述代码）：

.. code-block:: python

    place = fluid.CUDAPlace(0)
    startup_exe = fluid.Executor(place)
    startup_exe.run(train_startup)
    startup_exe.run(test_startup)

    trainer = fluid.ParallelExecutor(
        use_cuda=True, loss_name=train_loss.name, main_program=train_prog)

    tester = fluid.ParallelExecutor(
        use_cuda=True, share_vars_from=trainer, main_program=test_prog)

    train_reader.decorate_paddle_reader(
        paddle.reader.shuffle(paddle.batch(mnist.train(), 512), buf_size=8192))

    test_reader.decorate_paddle_reader(paddle.batch(mnist.test(), 512))

    for epoch_id in xrange(10):
        train_reader.start()
        try:
            while True:
                print 'train_loss', numpy.array(
                    trainer.run(fetch_list=[train_loss.name]))
        except fluid.core.EOFException:
            print 'End of epoch', epoch_id
            train_reader.reset()

        test_reader.start()
        try:
            while True:
                print 'test loss', numpy.array(
                    tester.run(fetch_list=[test_loss.name]))
        except fluid.core.EOFException:
            print 'End of testing'
            test_reader.reset()

具体步骤为：

1. 在每个epoch开始前，调用 :code:`start()` 方法启动PyReader对象；

2. 在每个epoch结束时， :code:`read_file` 抛出 :code:`fluid.core.EOFException` 异常，在捕获异常后调用 :code:`reset()` 方法重置PyReader对象的状态，以便启动下一轮的epoch。
