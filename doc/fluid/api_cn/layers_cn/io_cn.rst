=======
io
=======


.. _cn_api_fluid_layers_batch:

batch
-------------------------------

.. py:function:: paddle.fluid.layers.batch(reader, batch_size)

该层是一个reader装饰器。接受一个reader变量并添加``batching``装饰。读取装饰的reader，输出数据自动组织成batch的形式。

参数：
    - **reader** (Variable)-装饰有“batching”的reader变量
    - **batch_size** (int)-批尺寸

返回：装饰有``batching``的reader变量

返回类型：变量(Variable)

**代码示例**：

.. code-block:: python

    raw_reader = fluid.layers.io.open_files(filenames=['./data1.recordio',
                                               './data2.recordio'],
                                        shapes=[(3,224,224), (1,)],
                                        lod_levels=[0, 0],
                                        dtypes=['float32', 'int64'],
                                        thread_num=2,
                                        buffer_size=2)
    batch_reader = fluid.layers.batch(reader=raw_reader, batch_size=5)

    # 如果用raw_reader读取数据：
    #     data = fluid.layers.read_file(raw_reader)
    # 只能得到数据实例。
    #
    # 但如果用batch_reader读取数据：
    #     data = fluid.layers.read_file(batch_reader)
    # 每5个相邻的实例自动连接成一个batch。因此get('data')得到的是一个batch数据而不是一个实例。









.. _cn_api_fluid_layers_create_py_reader_by_data:

create_py_reader_by_data
-------------------------------

.. py:function:: paddle.fluid.layers.create_py_reader_by_data(capacity,feed_list,name=None,use_double_buffer=True)

创建一个 Python reader用于在python中提供数据,该函数将返回一个 ``reader`` 变量。

它的工作方式与 ``py_reader`` 非常相似，除了它的输入是一个 feed_list 而不是 ``shapes``、 ``dtypes`` 和 ``lod_level``

参数：
  - **capacity** (int) - 缓冲区容量由 :code:`py_reader` 维护
  - **feed_list** (list(Variable)) - 传输数据列表
  - **name** (basestring) - 前缀Python队列名称和 reader 名称。不定义时将自动生成名称。
  - **use_double_buffer** (bool) - 是否使用 double buffer

返回： Variable: 一种reader，我们可以从中获得输入数据。

**代码示例：**

 :code:`py_reader` 的基本用法如下所示：

.. code-block:: python
    
    import paddle
    import paddle.fluid as fluid
    import paddle.dataset.mnist as mnist

    def network(img, label):
        # 用户自定义网络。此处以一个简单的线性回归作为示例。
        predict = fluid.layers.fc(input=img, size=10, act='softmax')
        loss = fluid.layers.cross_entropy(input=predict, label=label)
        return fluid.layers.mean(loss)
    
    image = fluid.layers.data(name='image', shape=[1, 28, 28], dtypes='float32')
    label = fluid.layers.data(name='label', shape=[1], dtypes='int64')
    reader = fluid.layers.create_py_reader_by_data(capacity=64,
                                                   feed_list=[image, label])
    reader.decorate_paddle_reader(
        paddle.reader.shuffle(paddle.batch(mnist.train(), batch_size=5), buf_size=500))
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











.. _cn_api_fluid_layers_data:

data
-------------------------------

.. py:function:: paddle.fluid.layers.data(name, shape, append_batch_size=True, dtype='float32', lod_level=0, type=VarType.LOD_TENSOR, stop_gradient=True)

数据层(Data Layer)

该功能接受输入数据，判断是否需要以minibatch方式返回数据，然后使用辅助函数创建全局变量。该全局变量可由计算图中的所有operator访问。

这个函数的所有输入变量都作为本地变量传递给LayerHelper构造函数。

请注意，paddle在编译期间仅使用shape来推断网络中以下变量的形状。在运行期间，paddle不会检查所需数据的形状是否与此函数中的形状设置相匹配。

参数：
    - **name** (str)-函数名或函数别名
    - **shape** (list)-声明维度信息的list。如果 ``append_batch_size`` 为True且内部没有维度值为-1，则应将其视为每个样本的形状。 否则，应将其视为batch数据的形状。
    - **append_batch_size** (bool)-

        1.如果为真，则在维度shape的开头插入-1。
        例如，如果shape=[1],则输出shape为[-1,1]。这对在运行期间设置不同的batch大小很有用。

        2.如果维度shape包含-1，比如shape=[-1,1]。
        append_batch_size会强制变为为False（表示无效），因为PaddlePaddle不能在shape上设置一个以上的未知数。

    - **dtype** (np.dtype|VarType|str)-数据类型：float32,float_16,int等
    - **type** (VarType)-输出类型。默认为LOD_TENSOR
    - **lod_level** (int)-LoD层。0表示输入数据不是一个序列
    - **stop_gradient** (bool)-布尔类型，提示是否应该停止计算梯度

返回：全局变量，可进行数据访问

返回类型：变量(Variable)

**代码示例**：

.. code-block:: python

    data = fluid.layers.data(name='x', shape=[784], dtype='float32')










.. _cn_api_fluid_layers_double_buffer:

double_buffer
-------------------------------

.. py:function:: paddle.fluid.layers.double_buffer(reader, place=None, name=None)


生成一个双缓冲队列reader. 数据将复制到具有双缓冲队列的位置（由place指定），如果 ``place=none`` 则将使用executor执行的位置。

参数:
  - **reader** (Variable) – 需要wrap的reader
  - **place** (Place) – 目标数据的位置. 默认是executor执行样本的位置.
  - **name** (str) – Variable 的名字. 默认为None，不关心名称时也可以设置为None


返回： 双缓冲队列的reader


**代码示例**

..  code-block:: python

  import paddle.fluid as fluid
  reader = fluid.layers.open_files(filenames=['mnist.recordio'],
           shapes=[[-1, 784], [-1, 1]],
           dtypes=['float32', 'int64'])
  reader = fluid.layers.double_buffer(reader)
  img, label = fluid.layers.read_file(reader)












.. _cn_api_fluid_layers_load:

load
-------------------------------

.. py:function:: paddle.fluid.layers.load(out, file_path, load_as_fp16=None)

Load操作命令将从磁盘文件中加载LoDTensor/SelectedRows变量。

.. code-block:: python

    import paddle.fluid as fluid
    tmp_tensor = fluid.layers.create_tensor(dtype='float32')
    fluid.layers.load(tmp_tensor, "./tmp_tensor.bin")

参数：
    - **out** (Variable)-需要加载的LoDTensor或SelectedRows
    - **file_path** (STRING)-预从“file_path”中加载的变量Variable
    - **load_as_fp16** (BOOLEAN)-如果为真，张量首先进行加载然后类型转换成float16。如果为假，张量将直接加载，不需要进行数据类型转换。默认为false。

返回：None









.. _cn_api_fluid_layers_open_files:

open_files
-------------------------------

.. py:function:: paddle.fluid.layers.open_files(filenames, shapes, lod_levels, dtypes, thread_num=None, buffer_size=None, pass_num=1, is_test=None)

打开文件(Open files)

该函数获取需要读取的文件列表，并返回Reader变量。通过Reader变量，我们可以从给定的文件中获取数据。所有文件必须有名称后缀来表示它们的格式，例如，``*.recordio``。

参数：
    - **filenames** (list)-文件名列表
    - **shape** (list)-元组类型值列表，声明数据维度
    - **lod_levels** (list)-整形值列表，声明数据的lod层级
    - **dtypes** (list)-字符串类型值列表，声明数据类型
    - **thread_num** (None)-用于读文件的线程数。默认：min(len(filenames),cpu_number)
    - **buffer_size** (None)-reader的缓冲区大小。默认：3*thread_num
    - **pass_num** (int)-用于运行的传递数量
    - **is_test** (bool|None)-open_files是否用于测试。如果用于测试，生成的数据顺序和文件顺序一致。反之，无法保证每一epoch之间的数据顺序是一致的

返回：一个Reader变量，通过该变量获取文件数据

返回类型：变量(Variable)

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    reader = fluid.layers.io.open_files(filenames=['./data1.recordio',
                                            './data2.recordio'],
                                    shapes=[(3,224,224), (1,)],
                                    lod_levels=[0, 0],
                                    dtypes=['float32', 'int64'])

    # 通过reader, 可使用''read_file''层获取数据:
    image, label = fluid.layers.io.read_file(reader)









.. _cn_api_fluid_layers_Preprocessor:

Preprocessor
-------------------------------

.. py:class:: class paddle.fluid.layers.Preprocessor(reader, name=None)

reader变量中数据预处理块。

参数：
    - **reader** (Variable)-reader变量
    - **name** (str,默认None)-reader的名称

**代码示例**:

.. code-block:: python

    reader = fluid.layers.io.open_files(
        filenames=['./data1.recordio', './data2.recordio'],
        shapes=[(3, 224, 224), (1, )],
        lod_levels=[0, 0],
        dtypes=['float32', 'int64'])

    preprocessor = fluid.layers.io.Preprocessor(reader=reader)
    with preprocessor.block():
        img, lbl = preprocessor.inputs()
        img_out = img / 2
        lbl_out = lbl + 1
        preprocessor.outputs(img_out, lbl_out)
    data_file = fluid.layers.io.double_buffer(preprocessor())









.. _cn_api_fluid_layers_py_reader:

py_reader
-------------------------------

.. py:function:: paddle.fluid.layers.py_reader(capacity, shapes, dtypes, lod_levels=None, name=None, use_double_buffer=True)


创建一个由在Python端提供数据的reader

该layer返回一个Reader Variable。reader提供了 ``decorate_paddle_reader()`` 和 ``decorate_tensor_provider()`` 来设置Python generator作为数据源。更多细节请参考异步数据读取:ref:`user_guide_use_py_reader`，在c++端调用 ``Executor::Run()`` 时，来自generator的数据将被自动读取。与 ``DataFeeder.feed()`` 不同，数据读取进程和  ``Executor::Run()`` 进程可以使用 ``py_reader`` 并行运行。reader的 ``start()`` 方法应该在每次数据传递开始时调用，在传递结束和抛出  ``fluid.core.EOFException`` 后执行 ``reset()`` 方法。注意， ``Program.clone()`` 方法不能克隆 ``py_reader`` 。

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












.. _cn_api_fluid_layers_random_data_generator:

random_data_generator
-------------------------------

.. py:function:: paddle.fluid.layers.random_data_generator(low, high, shapes, lod_levels, for_parallel=True)

创建一个均匀分布随机数据生成器.

该层返回一个Reader变量。该Reader变量不是用于打开文件读取数据，而是自生成float类型的均匀分布随机数。该变量可作为一个虚拟reader来测试网络，而不需要打开一个真实的文件。

参数：
    - **low** (float)--数据均匀分布的下界
    - **high** (float)-数据均匀分布的上界
    - **shapes** (list)-元组数列表，声明数据维度
    - **lod_levels** (list)-整形数列表，声明数据
    - **for_parallel** (Bool)-若要运行一系列操作命令则将其设置为True

返回：Reader变量，可从中获取随机数据

返回类型：变量(Variable)

**代码示例**：

.. code-block:: python

    reader = fluid.layers.random_data_generator(
                                 low=0.0,
                                 high=1.0,
                                 shapes=[[3,224,224], [1]],
                                 lod_levels=[0, 0])
    # 通过reader, 可以用'read_file'层获取数据:
    image, label = fluid.layers.read_file(reader)









.. _cn_api_fluid_layers_read_file:

read_file
-------------------------------

.. py:function:: paddle.fluid.layers.read_file(reader)

执行给定的reader变量并从中获取数据

reader也是变量。可以为由fluid.layers.open_files()生成的原始reader或者由fluid.layers.double_buffer()生成的装饰变量，等等。

参数：
    - **reader** (Variable)-将要执行的reader

返回：从给定的reader中读取数据

返回类型: tuple（元组）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data_file = fluid.layers.open_files(
        filenames=['mnist.recordio'],
        shapes=[(-1, 748), (-1, 1)],
        lod_levels=[0, 0],
        dtypes=["float32", "int64"])
    data_file = fluid.layers.double_buffer(
        fluid.layers.batch(data_file, batch_size=64))
    input, label = fluid.layers.read_file(data_file)









.. _cn_api_fluid_layers_shuffle:

shuffle
-------------------------------

.. py:function:: paddle.fluid.layers.shuffle(reader, buffer_size)

创建一个特殊的数据读取器，它的输出数据会被重洗(shuffle)。由原始读取器创建的迭代器得到的输出将会被暂存到shuffle缓存区，其后
会对其进行重洗运算。shuffle缓存区的大小由参数 ``buffer_size`` 决定。

参数:
    - **reader** (callable) – 输出会被shuffle的原始reader
    - **buffer_size** (int) – 进行shuffle的buffer的大小

返回:其输出会被shuffle的一个reader（读取器）

返回类型:callable

**代码示例**：

.. code-block:: python

    raw_reader = fluid.layers.io.open_files(filenames=['./data1.recordio',
                                                   './data2.recordio'],
                                            shapes=[(3,224,224), (1,)],
                                            lod_levels=[0, 0],
                                            dtypes=['float32', 'int64'],
                                            thread_num=2,
                                            buffer_size=2)
    batch_reader = fluid.layers.batch(reader=raw_reader, batch_size=5)
    shuffle_reader = fluid.layers.shuffle(reader=batch_reader, buffer_size=5000)








