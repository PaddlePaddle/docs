.. _cn_api_fluid_layers_py_reader:

py_reader
-------------------------------


.. py:function:: paddle.fluid.layers.py_reader(capacity, shapes, dtypes, lod_levels=None, name=None, use_double_buffer=True)





创建一个在Python端提供数据的reader

该OP返回一个Reader Variable。该Reader提供了 ``decorate_paddle_reader()`` 和 ``decorate_tensor_provider()`` 来设置Python generator作为数据源，将数据源中的数据feed到Reader Variable。在c++端调用 ``Executor::Run()`` 时，来自generator的数据将被自动读取。与 ``DataFeeder.feed()`` 不同，数据读取进程和  ``Executor::Run()`` 进程可以使用 ``py_reader`` 并行运行。在每次数据传递开始时调用reader的 ``start()``，在传递结束和抛出  ``fluid.core.EOFException`` 异常后执行 ``reset()`` 。

注意：``Program.clone()`` （含义详见 :ref:`cn_api_fluid_Program` ）不能克隆 ``py_reader``，且 ``read_file`` （ ``read_file`` 含义详见 :ref:`cn_api_fluid_layers_read_file` ）调用需在声明 ``py_reader`` 的program block内。

参数
::::::::::::

  - **capacity** (int) –  ``py_reader`` 维护的缓冲区的容量数据个数。
  - **shapes** (list|tuple) – 一个列表或元组，shapes[i]是代表第i个数据shape，因此shape[i]也是元组或列表。
  - **dtypes** (list|tuple) – 一个string的列表或元组。为 ``shapes`` 对应元素的数据类型，支持bool，float16，float32，float64，int8，int16，int32，int64，uint8。
  - **lod_levels** (list|tuple) – lod_level的整型列表或元组
  - **name**  (str，可选) – 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为None。
  - **use_double_buffer** (bool) – 是否使用双缓冲区，双缓冲区是为了预读下一个batch的数据、异步CPU -> GPU拷贝。默认值为True。

返回
::::::::::::
reader，从reader中可以获取feed的数据，其dtype和feed的数据dtype相同。

返回类型
::::::::::::
Variable



代码示例
::::::::::::

1.py_reader 基本用法如下

..  code-block:: python

  import paddle
  import paddle.fluid as fluid
  import paddle.dataset.mnist as mnist

  def network(image, label):
      # 用户自定义网络，此处以softmax回归为例
      predict = fluid.layers.fc(input=image, size=10, act='softmax')
      return fluid.layers.cross_entropy(input=predict, label=label)
         
  reader = fluid.layers.py_reader(capacity=64,
                                  shapes=[(-1,1, 28, 28), (-1,1)],
                                  dtypes=['float32', 'int64'])
  reader.decorate_paddle_reader(
      paddle.reader.shuffle(paddle.batch(mnist.train(), batch_size=5),
                            buf_size=1000))

  img, label = fluid.layers.read_file(reader)
  loss = network(img, label) # 一些网络定义

  fluid.Executor(fluid.CUDAPlace(0)).run(fluid.default_startup_program())
  exe = fluid.ParallelExecutor(use_cuda=True, loss_name=loss.name)
  for epoch_id in range(10):
      reader.start()
          try:
              while True:
                  exe.run(fetch_list=[loss.name])
          except fluid.core.EOFException:
              reader.reset()

  fluid.io.save_inference_model(dirname='./model', 
                                feeded_var_names=[img.name, label.name],
                                target_vars=[loss], 
                                executor=fluid.Executor(fluid.CUDAPlace(0)))


2. 训练和测试应使用不同的名称创建两个不同的py_reader，例如：

..  code-block:: python

    import paddle
    import paddle.fluid as fluid
    import paddle.dataset.mnist as mnist

    def network(reader):
        img, label = fluid.layers.read_file(reader)
        # 用户自定义网络，此处以softmax回归为例
        predict = fluid.layers.fc(input=img, size=10, act='softmax')
        loss = fluid.layers.cross_entropy(input=predict, label=label)
        return fluid.layers.mean(loss)

    # 新建 train_main_prog 和 train_startup_prog
    train_main_prog = fluid.Program()
    train_startup_prog = fluid.Program()
    with fluid.program_guard(train_main_prog, train_startup_prog):
        # 使用 fluid.unique_name.guard() 实现与test program的参数共享
        with fluid.unique_name.guard():
            train_reader = fluid.layers.py_reader(capacity=64,
                                                shapes=[(-1, 1, 28, 28), (-1, 1)],
                                                dtypes=['float32', 'int64'],
                                                name='train_reader')
            train_reader.decorate_paddle_reader(
            paddle.reader.shuffle(paddle.batch(mnist.train(),
                                batch_size=5),
                                buf_size=500))
            train_loss = network(train_reader) # 一些网络定义
            adam = fluid.optimizer.Adam(learning_rate=0.01)
            adam.minimize(train_loss)

    # Create test_main_prog and test_startup_prog
    test_main_prog = fluid.Program()
    test_startup_prog = fluid.Program()
    with fluid.program_guard(test_main_prog, test_startup_prog):
        # 使用 fluid.unique_name.guard() 实现与train program的参数共享
        with fluid.unique_name.guard():
            test_reader = fluid.layers.py_reader(capacity=32,
                                                shapes=[(-1, 1, 28, 28), (-1, 1)],
                                                dtypes=['float32', 'int64'],
                                                name='test_reader')
            test_reader.decorate_paddle_reader(paddle.batch(mnist.test(), 512))
            test_loss = network(test_reader)

    fluid.Executor(fluid.CUDAPlace(0)).run(train_startup_prog)
    fluid.Executor(fluid.CUDAPlace(0)).run(test_startup_prog)

    train_exe = fluid.ParallelExecutor(use_cuda=True,
        loss_name=train_loss.name, main_program=train_main_prog)
    test_exe = fluid.ParallelExecutor(use_cuda=True,
        loss_name=test_loss.name, main_program=test_main_prog)
    for epoch_id in range(10):
        train_reader.start()
        try:
            while True:
                train_exe.run(fetch_list=[train_loss.name])
        except fluid.core.EOFException:
            train_reader.reset()

    test_reader.start()
    try:
        while True:
            test_exe.run(fetch_list=[test_loss.name])
    except fluid.core.EOFException:
        test_reader.reset()












