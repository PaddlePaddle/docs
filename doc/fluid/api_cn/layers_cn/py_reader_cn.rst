.. _cn_api_fluid_layers_py_reader:

py_reader
-------------------------------

.. py:function:: paddle.fluid.layers.py_reader(capacity, shapes, dtypes, lod_levels=None, name=None, use_double_buffer=True)


创建一个由在Python端提供数据的reader

该layer返回一个Reader Variable。reader提供了 ``decorate_paddle_reader()`` 和 ``decorate_tensor_provider()`` 来设置Python generator作为数据源。更多细节请参考 :ref:`user_guides_use_py_reader`，在c++端调用 ``Executor::Run()`` 时，来自generator的数据将被自动读取。与 ``DataFeeder.feed()`` 不同，数据读取进程和  ``Executor::Run()`` 进程可以使用 ``py_reader`` 并行运行。reader的 ``start()`` 方法应该在每次数据传递开始时调用，在传递结束和抛出  ``fluid.core.EOFException`` 后执行 ``reset()`` 方法。注意， ``Program.clone()`` 方法不能克隆 ``py_reader`` 。

参数:
  - **capacity** (int) –  ``py_reader`` 维护的缓冲区容量
  - **shapes** (list|tuple) –数据形状的元组或列表
  - **dtypes** (list|tuple) –  ``shapes`` 对应元素的数据类型
  - **lod_levels** (list|tuple) – lod_level的整型列表或元组
  - **name** (basestring) – python 队列的前缀名称和Reader 名称。不会自动生成。
  - **use_double_buffer** (bool) – 是否使用双缓冲

返回:    reader，从reader中可以获取feed的数据

返回类型: Variable



**代码示例**

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
      paddle.reader.shuffle(paddle.batch(mnist.train(), batch_size=5),buf_size=1000))

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

    fluid.io.save_inference_model(dirname='./model', feeded_var_names=[img.name, label.name],target_vars=[loss], executor=fluid.Executor(fluid.CUDAPlace(0)))


2.训练和测试应使用不同的名称创建两个不同的py_reader，例如：

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
      train_reader = fluid.layers.py_reader(capacity=64, shapes=[(-1, 1, 28, 28), (-1, 1)], dtypes=['float32', 'int64'], name='train_reader')
      train_reader.decorate_paddle_reader(
        paddle.reader.shuffle(paddle.batch(mnist.train(), batch_size=5), buf_size=500))
    train_loss = network(train_reader) # 一些网络定义
    adam = fluid.optimizer.Adam(learning_rate=0.01)
    adam.minimize(train_loss)

  # Create test_main_prog and test_startup_prog
  test_main_prog = fluid.Program()
  test_startup_prog = fluid.Program()
  with fluid.program_guard(test_main_prog, test_startup_prog):
    # 使用 fluid.unique_name.guard() 实现与train program的参数共享
    with fluid.unique_name.guard():
      test_reader = fluid.layers.py_reader(capacity=32, shapes=[(-1, 1, 28, 28), (-1, 1)], dtypes=['float32', 'int64'], name='test_reader')
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












