

.. _cn_api_fluid_layers_create_array:

create_array
>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.create_array(dtype)


创建LoDTensorArray数组。它主要用于实现RNN与array_write, array_read和While。

参数: 
    - **dtype** (int |float) — lod_tensor_array中存储元素的数据类型。

返回: lod_tensor_array， 元素数据类型为dtype。

返回类型: Variable。


**代码示例**

..  code-block:: python
  
  data = fluid.layers.create_array(dtype='float32')
  
  

.. _cn_api_fluid_layers_DynamicRNN:

DynamicRNN
>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.DynamicRNN(name=None)


动态RNN可以处理一批序列数据,每个样本序列的长度可以不同。这个API自动批量处理它们。

必须设置输入lod，请参考 ``lod_tensor``

**代码示例**

..  code-block:: python

	import paddle.fluid as fluid
	data = fluid.layers.data(name='sentence', dtype='int64', lod_level=1)
	embedding = fluid.layers.embedding(input=data, size=[65535, 32],
					    is_sparse=True)

	drnn = fluid.layers.DynamicRNN()
	with drnn.block():
		word = drnn.step_input(embedding)
	     	prev = drnn.memory(shape=[200])
	     	hidden = fluid.layers.fc(input=[word, prev], size=200, act='relu')
	     	drnn.update_memory(prev, hidden)  # set prev to hidden
	     	drnn.output(hidden)

	 # last是的最后一时间步，也是编码（encoding）得出的最终结果
	last = fluid.layers.sequence_last_step(drnn())


动态RNN将按照timesteps展开开序列。用户需要在with block中定义如何处理处理每个timestep。

memory用于缓存分段数据。memory的初始值可以是零，也可以是其他变量。

动态RNN可以将多个变量标记为其输出。使用drnn()获得输出序列。

.. py:method:: step_input(x)
  
    将序列标记为动态RNN输入。

参数:
    	- **x** - 输入序列	
	- **类型** - Variable
    	
返回:当前的输入序列中的timestep。

.. py:method:: static_input(x)

将变量标记为RNN输入。输入不会分散到timestep中。

参数:
    	- **x** - 输入变量
	- **类型** - Variable

返回:可以访问的RNN的输入变量,。

.. py:method:: block(*args, **kwds)

用户在RNN中定义operators的block。有关详细信息，请参阅class ``docstring`` 。

.. py:method:: memory(init=None, shape=None, value=0.0, need_reorder=False, dtype='float32')

为动态rnn创建一个memory 变量。
    
如果 ``init`` 不是None， ``memory`` 将由这个变量初始化。参数 ``need_reorder`` 用于将memory重新排序作为输入变量。当memory初始化依赖于输入样本时，应该将其设置为true。

**例如**

..  code-block:: python
  
  	import paddle.fluid as fluid
  	sentence = fluid.layers.data(
                 name='sentence', dtype='float32', shape=[32])
	boot_memory = fluid.layers.data(
                 name='boot', dtype='float32', shape=[10])

	drnn = fluid.layers.DynamicRNN()
	with drnn.block():
	     word = drnn.step_input(sentence)
	     memory = drnn.memory(init=boot_memory, need_reorder=True)
	     hidden = fluid.layers.fc(
			 input=[word, memory], size=10, act='tanh')
	     drnn.update_memory(ex_mem=memory, new_mem=hidden)
	     drnn.output(hidden)
	   
	rnn_output = drnn()



否则，如果已经设置 ``shape`` 、 ``value`` 、 ``dtype`` ，memory将被 ``value`` 初始化
  
..  code-block:: python
  
	import paddle.fluid as fluid

	sentence = fluid.layers.data(
			name='sentence', dtype='float32', shape=[32])

	drnn = fluid.layers.DynamicRNN()
	with drnn.block():
	    word = drnn.step_input(sentence)
	    memory = drnn.memory(shape=[10], dtype='float32', value=0)
	    hidden = fluid.layers.fc(
		    input=[word, memory], size=10, act='tanh')
	    drnn.update_memory(ex_mem=memory, new_mem=hidden)
	    drnn.output(hidden)
	rnn_output = drnn()


参数：
    - **init** (Variable|None) – 初始化的Variable
    - **shape** (list|tuple) – memory shape. 注意形状不包含batch的大小
    - **value** (float) – 初始化的值
    - **need_reorder** (bool) – memory初始化依赖于输入样本时设置为True
    - **dtype** (str|numpy.dtype) – 初始化memory的数据类型

返回：memory Variable


.. py:method:: update_memory(ex_mem, new_mem)

将内存从 ``ex_mem`` 更新到 ``new_mem`` 。注意， ``ex_mem`` 和 ``new_mem`` 的 ``shape`` 和数据类型必须相同。

参数：
	- **ex_mem**（memory Variable）-  memory 变量（Variable） 
	- **new_mem**（memory Variable）- RNN块中生成的平坦变量（plain  variable）

返回：None


.. py:method:: output(*outputs)

标记RNN输出变量。

参数:
    - outputs - 输出变量。

返回:None
 
 
.. _cn_api_fluid_layers_StaticRNN:

StaticRNN
>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.StaticRNN(name=None)


用于创建static RNN。RNN将有自己的参数，比如输入、输出、memory、状态和长度。

.. py:method:: memory(init=None, shape=None, batch_ref=None, init_value=0.0, init_batch_dim_idx=0, ref_batch_dim_idx=1)

参数：
    - **init** - boot memory，如果没有设置，则必须提供一个shape
    - **shape** - boot memory的形状
    - **batch_ref** - batch引用
    - **init_value** - boot memory的初始化值
    - **init_batch_dim_idx** - init维度中的batch大小的索引
    - **ref_batch_dim_idx** - batch_ref维度中的batch大小的索引



 
.. _cn_api_fluid_layers_shuffle:

shuffle
>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.shuffle(reader, buffer_size)

使用python装饰器用shuffle 装饰 reader

参数:
    - **reader**(Variable) – 用shuffle装饰的reader
    - **buffer_size** (int) – reader中buffer的大小

返回:用shuffle装饰后的reader

返回类型:Variable


.. _cn_api_fluid_layers_double_buffer:

double_buffer
>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.double_buffer(reader, place=None, name=None)


生成一个双缓冲队列reader. 数据将复制到具有双缓冲队列的位置（由place指定），如果 ``place=none`` 则将使用executor执行的位置。

参数:
  - **reader** (Variable) – 需要wrap的reader
  - **place** (Place) – 目标数据的位置. 默认是executor执行样本的位置.
  - **name** (str) – Variable 的名字. 默认为None，不关心名称时也可以设置为None


返回： 双缓冲队列的reader


**代码示例**

..  code-block:: python

	reader = fluid.layers.open_files(filenames=['somefile'],
					 shapes=[[-1, 784], [-1, 1]],
					 dtypes=['float32', 'int64'])
	reader = fluid.layers.double_buffer(reader)
	img, label = fluid.layers.read_file(reader)




.. _cn_api_fluid_layers_py_reader:

py_reader
>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.py_reader(capacity, shapes, dtypes, lod_levels=None, name=None, use_double_buffer=True)


创建一个由在Python端提供数据的reader

该layer返回一个Reader Variable。reader提供了 ``decorate_paddle_reader()`` 和 ``decorate_tensor_provider()`` 来设置Python generator，作为Python端的数据源。在c++端调用 ``Executor::Run()`` 时，来自generator的数据将被自动读取。与 ``DataFeeder.feed()`` 不同，数据读取进程和  ``Executor::Run()`` 进程可以使用 ``py_reader`` 并行运行。reader的 ``start()`` 方法应该在每次数据传递开始时调用，在传递结束和抛出  ``fluid.core.EOFException`` 后执行 ``reset()`` 方法。注意， ``Program.clone()`` 方法不能克隆 ``py_reader`` 。

参数:	
  - **capacity** (int) –  ``py_reader`` 维护的缓冲区容量
  - **shapes** (list|tuple) –数据形状的元组或列表.
  - **dtypes** (list|tuple) –  ``shapes`` 对应元素的数据类型
  - **lod_levels** (list|tuple) – lod_level的整型列表或元组
  - **name** (basestring) – python 队列的前缀名称和Reader 名称。不会自动生成。
  - **use_double_buffer** (bool) – 是否使用双缓冲

返回:    reader，从reader中可以获取feed的数据

返回类型:	Variable
	


**代码示例**

1. py_reader 基本使用如下代码

..  code-block:: python

	import paddle.v2
	import paddle.fluid as fluid
	import paddle.dataset.mnist as mnist

	reader = fluid.layers.py_reader(capacity=64,
					shapes=[(-1,3,224,224), (-1,1)],
					dtypes=['float32', 'int64'])
	reader.decorate_paddle_reader(
	    paddle.v2.reader.shuffle(paddle.batch(mnist.train())

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



**代码示例**

2. 训练和测试应使用不同的名称创建两个不同的py_reader，例如：

..  code-block:: python

	import paddle.v2
	import paddle.fluid as fluid
	import paddle.dataset.mnist as mnist

	def network(reader):
	    img, label = fluid.layers.read_file(reader)
	    # 此处我们省略了一些网络定义
	    return loss

	train_reader = fluid.layers.py_reader(capacity=64,
					      shapes=[(-1,3,224,224), (-1,1)],
					      dtypes=['float32', 'int64'],
					      name='train_reader')
	train_reader.decorate_paddle_reader(
	    paddle.v2.reader.shuffle(paddle.batch(mnist.train())

	test_reader = fluid.layers.py_reader(capacity=32,
					     shapes=[(-1,3,224,224), (-1,1)],
					     dtypes=['float32', 'int64'],
					     name='test_reader')
	test_reader.decorate_paddle_reader(paddle.batch(mnist.test(), 512))

	# 新建 train_main_prog 和 train_startup_prog
	train_main_prog = fluid.Program()
	train_startup_prog = fluid.Program()
	with fluid.program_guard(train_main_prog, train_startup_prog):
	    # 使用 fluid.unique_name.guard() 实现与test program的参数共享
	    with fluid.unique_name.guard():
		train_loss = network(train_reader) # 一些网络定义
		adam = fluid.optimizer.Adam(learning_rate=0.01)
		adam.minimize(loss)

	# Create test_main_prog and test_startup_prog
	test_main_prog = fluid.Program()
	test_startup_prog = fluid.Program()
	with fluid.program_guard(test_main_prog, test_startup_prog):
	    # 使用 fluid.unique_name.guard() 实现与train program的参数共享
	    with fluid.unique_name.guard():
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




.. _cn_api_fluid_layers_log:

log
>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.log(x, name=None)


给定输入张量，计算其每个元素的自然对数

.. math::
                  \\Out=ln(x)\\
 

参数:

  - **x** (Variable) – 输入张量
  - **name** (str|None, default None) – 该layer的名称，如果为None，自动命名

返回：给定输入张量计算自然对数

返回类型:	变量（variable）


**代码示例**

..  code-block:: python

  output = fluid.layers.log(x)
