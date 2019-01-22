.. _user_guide_use_py_reader:

#########################################
Use PyReader to read train and test data
##########################################

Paddle Fluid support PyReader,implementing feeding data from Python to C++.Differenting from :ref:`user_guide_use_numpy_array_as_train_data` , the process of loading to Python is asynchronous with the process of reading data :code:`Executor::Run()` with PyReader
and is able to work with :code:`double_buffer_reader` to upgrade the performance of reading data.

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

capacity is buffer size of PyReader;
shapes is the size of parameters of batch(such as image and label in picture classification task);
dtypes is data type of parameters of batch;
name is name of PyReader;
use_double_buffer is True by default,expressed by :code:`double_buffer_reader` 。

If there are different PyReader objects created,the names of objects should be unique. Usually,you have to create two different PyReader objects in training and testing period.For example.In the same task,PyReader objects in training and testing period are created as follows:

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

Note: You could not copy PyReader object with :code:`Program.clone()` so you have to create PyReader objects in training and testing period with the method mentioned above

Because you could not copy PyReader with :code:`Program.clone()` so you have to get parameters in training and testing period shared with :code:`fluid.unique_name.guard()` .

Details are as follows:

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.dataset.mnist as mnist
    import paddle.v2

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
PyReader provide :code:`decorate_tensor_provider` and :code:`decorate_paddle_reader` ,both of which receieve Python :code:`generator` as data source.The difference is:

1. :code:`decorate_tensor_provider` : everytime :code:`generator` generates a  :code:`list` or :code:`tuple` ,each element of :code:`list` 或 :code:`tuple` is :code:`LoDTensor` or Numpy array,and :code:`LoDTensor` or :code:`shape` of Numpy array must be the same with :code:`shapes` stated while PyReader is created.


2. :code:`decorate_paddle_reader` : everytime :code:`generator` generates a :code:`list` or :code:`tuple` ,each element of :code:`list` 或 :code:`tuple` is Numpy array,but :code:`shape` of Numpy array have not to be the same with :code:`shape` stated while PyReader is created. :code:`decorate_paddle_reader` will :code:`reshape` :code:`shape` of Numpy array

Train and test model with PyReader
##################################

Details are as follows（the left part of code above）:

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
        paddle.v2.reader.shuffle(paddle.batch(mnist.train(), 512), buf_size=8192))

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

1. Before the start of every epoch,call :code:`start()` to invoke PyReader;

2. At the end of every epoch, :code:`read_file` throw exception :code:`fluid.core.EOFException` . Call :code:`reset()` after catching up exception to reset the state of PyReader in order to start next epoch.
