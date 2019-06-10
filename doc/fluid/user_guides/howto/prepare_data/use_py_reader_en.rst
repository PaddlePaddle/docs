.. _user_guide_use_py_reader_en:

############################################
Use PyReader to read training and test data
############################################

Paddle Fluid supports PyReader, which implements feeding data from Python to C++. Different from :ref:`user_guide_use_numpy_array_as_train_data_en` , the process of loading data to Python is asynchronous with the process of :code:`Executor::Run()` reading data when PyReader is in use.
Moreover, PyReader is able to work with :code:`double_buffer_reader` to upgrade the performance of reading data.

Create PyReader Object
################################

You can create PyReader object as follows:

.. code-block:: python

    import paddle.fluid as fluid

    py_reader = fluid.layers.py_reader(capacity=64,
                                       shapes=[(-1,3,224,224), (-1,1)],
                                       dtypes=['float32', 'int64'],
                                       name='py_reader',
                                       use_double_buffer=True)

In the code, ``capacity`` is buffer size of PyReader; 
``shapes`` is the size of parameters in the batch (such as image and label in picture classification task); 
``dtypes`` is data type of parameters in the batch; 
``name`` is name of PyReader instance; 
``use_double_buffer`` is True by default, which means :code:`double_buffer_reader` is used.

To create some different PyReader objects (Usually, you have to create two different PyReader objects for training and testing phase), the names of objects must be different. For example, In the same task, PyReader objects in training and testing period are created as follows:

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

Note: You could not copy PyReader object with :code:`Program.clone()` so you have to create PyReader objects in training and testing phase with the method mentioned above

Because you could not copy PyReader with :code:`Program.clone()` so you have to share the parameters of training phase with testing phase through :code:`fluid.unique_name.guard()` .

Details are as follows:

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

Configure data source of PyReader objects
##########################################
PyReader provides :code:`decorate_tensor_provider` and :code:`decorate_paddle_reader` , both of which receieve Python :code:`generator` as data source.The difference is:

1. :code:`decorate_tensor_provider` :  :code:`generator` generates a  :code:`list` or :code:`tuple` each time, with each element of :code:`list` or :code:`tuple` being :code:`LoDTensor` or Numpy array, and :code:`LoDTensor` or :code:`shape` of Numpy array must be the same as :code:`shapes` stated while PyReader is created.


2. :code:`decorate_paddle_reader` :  :code:`generator` generates a :code:`list` or :code:`tuple` each time, with each element of :code:`list` or :code:`tuple` being Numpy array,but the :code:`shape` of Numpy array doesn't have to be the same as :code:`shape` stated while PyReader is created. :code:`decorate_paddle_reader` will :code:`reshape` Numpy array internally.

example usage：

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

Train and test model with PyReader
##################################

Details are as follows（the remaining part of the code above）:

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

Specific steps are as follows:

1. Before the start of every epoch, call :code:`start()` to invoke PyReader;

2. At the end of every epoch, :code:`read_file` throws exception :code:`fluid.core.EOFException` . Call :code:`reset()` after catching up exception to reset the state of PyReader in order to start next epoch.
