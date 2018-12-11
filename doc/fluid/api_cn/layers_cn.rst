###################
fluid.layers
###################


============
 control_flow 
============


.. _cn_api_fluid_layers_array_length:

array_length
-------------------------------

.. py:function:: paddle.fluid.layers.array_length(array)

**得到输入LoDTensorArray的长度**

此功能用于查找输入数组LOD_TENSOR_ARRAY的长度。  

相关API:
    - :ref:`cn_api_fluid_layers_array_read`
    - :ref:`cn_api_fluid_layers_array_write`
    - :ref:`cn_api_fluid_layers_While`

参数：
    - **array** (LOD_TENSOR_ARRAY)-输入数组，用来计算数组长度

返回：输入数组LoDTensorArray的长度

返回类型：变量（Variable）

**代码示例**:

.. code-block:: python

    tmp = fluid.layers.zeros(shape=[10], dtype='int32')
    i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
    arr = fluid.layers.array_write(tmp, i=i)
    arr_len = fluid.layers.array_length(arr)









.. _cn_api_fluid_layers_array_read:

array_read
-------------------------------

.. py:function:: paddle.fluid.layers.array_read(array,i)

此函数用于读取数据，数据以LOD_TENSOR_ARRAY数组的形式读入

::


    Given:
        array = [0.6,0.1,0.3,0.1]
    And:
        I=2
    Then:
        output = 0.3

参数：
    - **array** (Variable|list)-输入张量，存储要读的数据
    - **i** (Variable|list)-输入数组中数据的索引

返回：张量类型的变量，已有数据写入

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    tmp = fluid.layers.zeros(shape=[10],dtype='int32')
    i = fluid.layers.fill_constant(shape=[1],dtype='int64',value=10)
    arr = layers.array_read(tmp,i=i)









.. _cn_api_fluid_layers_array_write:
    
array_write
-------------------------------

.. py:function:: paddle.fluid.layers.array_write(x, i, array=None)


该函数将给定的输入变量（即 ``x`` ）写入一个作为输出的 ``LOD_TENSOR_ARRAY`` 变量的某一指定位置中，
这一位置由数组下标(即 ``i`` )指明。 如果 ``LOD_TENSOR_ARRAY`` (即 ``array`` )未指定（即为None值）， 一个新的 ``LOD_TENSOR_ARRAY`` 将会被创建并作为结果返回。

参数:
    - **x** (Variable|list) – 待从中读取数据的输入张量(tensor)
    - **i** (Variable|list) – 输出结果 ``LOD_TENSOR_ARRAY`` 的下标, 该下标指向输入张量 ``x`` 写入输出数组的位置
    - **array** (Variable|list) – 会被输入张量 ``x`` 写入的输出结果 ``LOD_TENSOR_ARRAY`` 。如果该项值为None， 一个新的 ``LOD_TENSOR_ARRAY`` 将会被创建并作为结果返回
 
返回:	输入张量 ``x`` 所写入的输出结果 ``LOD_TENSOR_ARRAY``  

返回类型:	变量（Variable）

**代码示例**

..  code-block:: python

  tmp = fluid.layers.zeros(shape=[10], dtype='int32')
  i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
  arr = layers.array_write(tmp, i=i)











.. _cn_api_fluid_layers_create_array:

create_array
-------------------------------

.. py:function:: paddle.fluid.layers.create_array(dtype)


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
-------------------------------

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

.. note::
    目前不支持在DynamicRNN中任何层上配置 is_sparse = True

.. py:method:: step_input(x)
  
    将序列标记为动态RNN输入。

参数:
    	- **x** (Variable) - 输入序列	
	
    	
返回:当前的输入序列中的timestep。

.. py:method:: static_input(x)

将变量标记为RNN输入。输入不会分散到timestep中。

参数:
    	- **x** (Variable) - 输入序列

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
	- **ex_mem** （memory Variable）-  memory 变量（Variable） 
	- **new_mem** （memory Variable）- RNN块中生成的平坦变量（plain  variable）

返回：None


.. py:method:: output(*outputs)

标记RNN输出变量。

参数:
    - **\*outputs** - 输出变量。

返回:None
 
 








.. _cn_api_fluid_layers_equal:

equal
-------------------------------

.. py:function:: paddle.fluid.layers.equal(x,y,cond=None,**ignored)

**equal**
该层返回 :math:`x==y` 按逐元素运算而得的真值。

参数：
    - **x** (Variable)-equal的第一个操作数
    - **y** (Variable)-equal的第二个操作数
    - **cond** (Variable|None)-输出变量（可选），用来存储equal的结果

返回：张量类型的变量，存储equal的输出结果 

返回类型：变量（Variable） 

**代码示例**: 

.. code-block:: python

    less = fluid.layers.equal(x=label,y=limit)









.. _cn_api_fluid_layers_IfElse:

IfElse
-------------------------------

.. py:function:: paddle.fluid.layers.IfElse(cond, name=None)

if-else控制流。  

参数：
    - **cond** (Variable)-用于比较的条件
    - **Name** (str,默认为空（None）)-该层名称

**代码示例**：

.. code-block:: python

    limit = fluid.layers.fill_constant_batch_size_like(
        input=label, dtype='int64', shape=[1], value=5.0)
    cond = fluid.layers.less_than(x=label, y=limit)
    ie = fluid.layers.IfElse(cond)
    with ie.true_block():
        true_image = ie.input(image)
        hidden = fluid.layers.fc(input=true_image, size=100, act='tanh')
        prob = fluid.layers.fc(input=hidden, size=10, act='softmax')
        ie.output(prob)

    with ie.false_block():
        false_image = ie.input(image)
        hidden = fluid.layers.fc(
            input=false_image, size=200, act='tanh')
        prob = fluid.layers.fc(input=hidden, size=10, act='softmax')
        ie.output(prob)
    prob = ie()









.. _cn_api_fluid_layers_increment:
  
increment
-------------------------------
  
.. py:function:: paddle.fluid.layers.increment(x, value=1.0, in_place=True)

   
该函数为输入 ``x`` 增加 ``value`` 大小, ``value`` 即函数中待传入的参数。该函数默认直接在原变量 ``x`` 上进行运算。

.. note::
    ``x`` 中元素个数必须为1

参数:
    - **x** (Variable|list) – 含有输入值的张量(tensor)
    - **value** (float) – 需要增加在 ``x`` 变量上的值
    - **in_place** (bool) – 判断是否在x变量本身执行操作，True原地执行，False时，返回增加后的副本

返回： 每个元素增加后的对象

返回类型：变量(variable)

**代码示例**

..  code-block:: python
  
    data = fluid.layers.data(name='data', shape=[1], dtype='float32',
                         append_batch_size=False)
    data = fluid.layers.increment(x=data, value=3.0, in_place=True)
 
 
 








.. _cn_api_fluid_layers_is_empty:

is_empty
-------------------------------

.. py:function:: paddle.fluid.layers.is_empty(x, cond=None, **ignored)

测试变量是否为空

参数：
    - **x** (Variable)-测试的变量
    - **cond** (Variable|None)-输出参数。返回给定x的测试结果，默认为空（None）

返回：布尔类型的标量。如果变量x为空则值为真

返回类型：变量（Variable）

抛出异常：``TypeError``-如果input不是变量或cond类型不是变量

**代码示例**：

.. code-block:: python

    res = fluid.layers.is_empty(x=input)
    # or:
    fluid.layers.is_empty(x=input, cond=res)









.. _cn_api_fluid_layers_less_than:

less_than
-------------------------------

.. py:function:: paddle.fluid.layers.less_than(x, y, force_cpu=None, cond=None, **ignored)


该函数按元素出现顺序依次在X,Y上操作，并返回 ``Out`` ，它们三个都是n维tensor（张量）。
其中，X、Y可以是任何类型的tensor，Out张量的各个元素可以通过 :math:`Out=X<Y` 计算得出。

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    less = fluid.layers.less_than(x=label, y=limit)

参数：  
    - **x** (Variable) – ``less_than`` 运算的左操作数
    - **y** (Variable) – ``less_than`` 运算的右操作数
    - **force_cpu** (BOOLEAN) – 值True则强制将输出变量写入CPU内存中。否则，将其写入目前所在的运算设备上。默认为True
    - **cond** (Variable|None) – 可选的用于存储 ``less_than`` 输出结果的变量，为None则由函数自动生成Out变量


返回：	n维bool型tensor，其中各个元素可以通过 *Out=X<Y* 计算得出













.. _cn_api_fluid_layers_Print:

Print
-------------------------------

.. py:function:: paddle.fluid.layers.Print(input, first_n=-1, message=None, summarize=-1, print_tensor_name=True, print_tensor_type=True, print_tensor_shape=True, print_tensor_lod=True, print_phase='both')

**Print操作命令**

该操作命令创建一个打印操作，打印正在访问的张量。

封装传入的张量，以便无论何时访问张量，都会打印信息message和张量的当前值。

参数：
    - **input** (Variable)-将要打印的张量
    - **summarize** (int)-打印张量中的元素数目，如果值为-1则打印所有元素
    - **message** (str)-字符串类型消息，作为前缀打印
    - **first_n** (int)-只记录first_n次数
    - **print_tensor_name** (bool)-打印张量名称
    - **print_tensor_type** (bool)-打印张量类型
    - **print_tensor_shape** (bool)-打印张量维度
    - **print_tensor_lod** (bool)-打印张量lod
    - **print_phase** (str)-打印的阶段，包括 ``forward`` , ``backward`` 和 ``both`` .若设置为 ``backward`` 或者 ``both`` ,则打印输入张量的梯度。

返回：输出张量，和输入张量同样的数据

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    value = some_layer(...)
    Print(value, summarize=10,
    message="The content of some_layer: ")









.. _cn_api_fluid_layers_reorder_lod_tensor_by_rank:

reorder_lod_tensor_by_rank
-------------------------------

.. py:function:: paddle.fluid.layers.reorder_lod_tensor_by_rank(x, rank_table)


函数参数 ``X`` 是由多个序列(sequence)组成的的一个batch（数据批）。``rank_table`` 存储着batch中序列的重新排列规则。
该operator（算子）根据 ``rank_table`` 中提供的规则信息来实现对 ``X`` 的重新排列。


::
	
  例如:
 
  假设在 RankTable 中存储的序列索引为 [3,0,2,1]， X 将会被这样被重新排列：
  X 中的第四个序列（即索引为3的序列，后面以此类推）会变成排列后的batch中的第一个，紧接着就是原来batch中的第一个元素，第三个元素，和第二个元素。
  简言之，若有原batch：X = [Seq0, Seq1, Seq2, Seq3] 且 RankTable 中的索引为 [3,0,2,1]，那么输出即为 Out = [Seq3, Seq0, Seq2, Seq1] ，它携带着新的LoD信息。	
  如果 X 的LoD信息是空的，这表明 X 不是序列型数据。这和由多个定长为1的序列组成的batch是相同的情况。此时，该函数将对 X 中的切片（slice） 在第一轴(axis)上按 rank_table 里的规则加以排列。
  例如，现有 X = [Slice0, Slice1, Slice2, Slice3] ，并且它LoD信息为空，在 RankTable 索引为[3, 0, 2, 1]。则 Out = [Slice3, Slice0, Slice2, Slice1] ，并且不在其中追加LoD信息。

注意，该operator对 ``X`` 进行的排序所依据的 ``LoDRankTable`` 不一定是在 ``X`` 的基础上得出来的。它可以由
其他不同的序列batch得出，并由该operator依据这个 ``LoDRankTable`` 来对  ``X`` 排序。

参数：   
    - **x** (LoDTensor)-待根据提供的 ``RankTable`` 进行排序的LoD tensor
    - **rank_table** (LoDRankTable)- ``X`` 重新排序的依据规则表


返回：	重新排列后的LoDTensor

返回类型:	LoDTensor














.. _cn_api_fluid_layers_StaticRNN:

StaticRNN
-------------------------------

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



 








.. _cn_api_fluid_layers_Switch:

Switch
-------------------------------

.. py:function:: paddle.fluid.layers.Switch (name=None)

Switch类实现的功能十分类似if-elif-else。它可以在学习率调度器(learning rate scheduler)中调整学习率。
:: 
  语义上，
      1. switch控制流挨个检查cases
      2. 各个case的条件是一个布尔值(boolean)，它是一个标量(scalar)变量
      3. 它将执行第一个匹配的case后面的分支，如果没有匹配的case，但若存在一个default case,则会执行default case后面的语句
      4. 一旦匹配了一个case,它降会执行这个case所对应的分支，且仅此分支。

**代码示例**

..  code-block:: python
    
    lr = fluid.layers.tensor.create_global_var(
        shape=[1],
        value=0.0,
        dtype='float32',
        persistable=True,
        name="learning_rate")
    one_var = tensor.fill_constant(
        shape=[1], dtype='float32', value=1.0)
    two_var = tensor.fill_constant(
        shape=[1], dtype='float32', value=2.0)

    with fluid.layers.control_flow.Switch() as switch:
        with switch.case(global_step == zero_var):
            fluid.layers.tensor.assign(input=one_var, output=lr)
        with switch.default():
            fluid.layers.tensor.assign(input=two_var, output=lr)
 
.. py:method:: case(condition)

为该condition（情况，条件）建立新的block（块）。
  
  
.. py:method:: default()

为该switch建立default case。












.. _cn_api_fluid_layers_While:

While
-------------------------------

.. py:function:: paddle.fluid.layers.While (cond, is_test=False, name=None)


该类用于实现while循环控制功能。


参数：  
		- **cond** (Variable) – 用于比较的条件
		- **is_test** (bool) – 用于表明是不是在测试阶段执行
		- **name** (str) - 该层的命名
 
**代码示例**

..  code-block:: python

  d0 = layers.data("d0", shape=[10], dtype='float32')
  data_array = layers.array_write(x=d0, i=i)
  array_len = layers.fill_constant(shape=[1],dtype='int64', value=3)
  
  cond = layers.less_than(x=i, y=array_len)
  while_op = layers.While(cond=cond)
  with while_op.block():
      d = layers.array_read(array=data_array, i=i)
      i = layers.increment(x=i, in_place=True)
      layers.array_write(result, i=i, array=d)
      layers.less_than(x=i, y=array_len, cond=cond)











============
 io 
============


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
                                        shapes=[(3,224,224), (1)],
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

    import paddle.fluid as fluid
    import paddle.dataset.mnist as mnist
    
    image = fluid.layers.data(name='image', shape=[3,224,224], dtypes='float32')
    label = fluid.layers.data(name='label', shape=[1], dtypes='int64')
    reader = fluid.layers.create_py_reader_by_data(capacity=64, feed_list=[image, label])
    reader.decorate_paddle_reader(
        paddle.reader.shuffle(paddle.batch(mnist.train())
    img, label = fluid.layers.read_file(reader)
    loss = network(img, label) # some network definition

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

参数：
    - **name** (str)-函数名或函数别名
    - **shape** (list)-声明维度的元组
    - **append_batch_size** (bool)-

        1.如果为真，则在维度shape的开头插入-1
        “如果shape=[1],则输出shape为[-1,1].”

        2.如果维度shape包含-1，比如shape=[-1,1],
        “append_batch_size则为False（表示无效）”

    - **dtype** (basestring)-数据类型：float32,float_16,int等
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

	reader = fluid.layers.open_files(filenames=['somefile'],
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

    reader = fluid.layers.io.open_files(filenames=['./data1.recordio',
                                            './data2.recordio'],
                                    shapes=[(3,224,224), (1)],
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

1.py_reader 基本使用如下代码

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





2.训练和测试应使用不同的名称创建两个不同的py_reader，例如：

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

**代码示例**：

.. code-block:: python

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
    - **reader** (Variable) – 用shuffle装饰的reader
    - **buffer_size** (int) – reader中buffer的大小

返回:其输出会被重洗的一个reader（读取器）

返回类型:Variable












============
 nn 
============


.. _cn_api_fluid_layers_add_position_encoding:

add_position_encoding
-------------------------------

.. py:function:: paddle.fluid.layers.add_position_encoding(input, alpha, beta, name=None)

**添加位置编码层**

接受形状为[N×M×P]的三维输入张量，并返回一个形为[N×M×P]的输出张量，且输出张量具有位置编码值。

可参考论文: `Attention Is All You Need <http://arxiv.org/pdf/1706.03762.pdf>`_ 

.. math::

  PE(pos, 2i) &= \sin{(pos / 10000^{2i / P})}\\
  PE(pos, 2i + 1) &= \cos{(pos / 10000^{2i / P})}\\
  Out(:, pos, i) &= \alpha * input(:, pos, i) + \beta * PE(pos, i)
	
其中:
    - PE(pos, 2i): 偶数位置上数字的增量
    - PE(pos, 2i + 1): 奇数位置上数字的增量

返回： Variable: 具有位置编码的三维形状张量[N×M×P]

**代码示例：**

.. code-block:: python

	position_tensor = fluid.layers.add_position_encoding(input=tensor)











.. _cn_api_fluid_layers_affine_channel:

affine_channel
-------------------------------

.. py:function:: paddle.fluid.layers.affine_channel(x, scale=None, bias=None, data_layout='NCHW', name=None)

对输入的每个 channel 应用单独的仿射变换。用于将空间批处理范数替换为其等价的固定变换。

输入也可以是二维张量，并在二维应用仿射变换。

参数：
	- **x** (Variable):特征图输入可以是一个具有NCHW阶或NHWC阶的4D张量。它也可以是二维张量和应用于第二维度的仿射变换。
	- **scale** (Variable): 形状为(C)的一维输入，第C个元素为输入的第C通道仿射变换的尺度因子。
	- **bias** (Variable):形状为(C)的一维输入，第C个元素是输入的第C个通道的仿射变换的偏置。
	- **data_layout** (string, default NCHW): NCHW 或 NHWC，如果输入是一个2D张量，可以忽略该参数
	- **name** (str, default None): 此层的名称

返回： out (Variable): 与x具有相同形状和数据布局的张量。









.. _cn_api_fluid_layers_affine_grid:

affine_grid
-------------------------------

.. py:function:: paddle.fluid.layers.affine_grid(theta, out_shape, name=None)

它使用仿射变换的参数生成(x,y)坐标的网格，这些参数对应于一组点，在这些点上，输入特征映射应该被采样以生成转换后的输出特征映射。



.. code-block:: text
        
        * 例 1:
          给定:
              theta = [[[x_11, x_12, x_13]
                        [x_14, x_15, x_16]]
                       [[x_21, x_22, x_23]
                        [x_24, x_25, x_26]]]
              out_shape = [2, 3, 5, 5]
          
          Step 1:
              
              根据out_shape生成标准化坐标

              归一化坐标的值在-1和1之间
              
              归一化坐标的形状为[2,H, W]，如下所示:
              
              C = [[[-1.  -1.  -1.  -1.  -1. ]
                    [-0.5 -0.5 -0.5 -0.5 -0.5]
                    [ 0.   0.   0.   0.   0. ]
                    [ 0.5  0.5  0.5  0.5  0.5]
                    [ 1.   1.   1.   1.   1. ]]
                   [[-1.  -0.5  0.   0.5  1. ]
                    [-1.  -0.5  0.   0.5  1. ]
                    [-1.  -0.5  0.   0.5  1. ]
                    [-1.  -0.5  0.   0.5  1. ]
                    [-1.  -0.5  0.   0.5  1. ]]]
              
              C[0]是高轴坐标，C[1]是宽轴坐标。

          Step2:
              
              将C转换并重组成形为[H * W, 2]的张量,并追加到最后一个维度

              我们得到:

              C_ = [[-1.  -1.   1. ]
                    [-0.5 -1.   1. ]
                    [ 0.  -1.   1. ]
                    [ 0.5 -1.   1. ]
                    [ 1.  -1.   1. ]
                    [-1.  -0.5  1. ]
                    [-0.5 -0.5  1. ]
                    [ 0.  -0.5  1. ]
                    [ 0.5 -0.5  1. ]
                    [ 1.  -0.5  1. ]
                    [-1.   0.   1. ]
                    [-0.5  0.   1. ]
                    [ 0.   0.   1. ]
                    [ 0.5  0.   1. ]
                    [ 1.   0.   1. ]
                    [-1.   0.5  1. ]
                    [-0.5  0.5  1. ]
                    [ 0.   0.5  1. ]
                    [ 0.5  0.5  1. ]
                    [ 1.   0.5  1. ]
                    [-1.   1.   1. ]
                    [-0.5  1.   1. ]
                    [ 0.   1.   1. ]
                    [ 0.5  1.   1. ]
                    [ 1.   1.   1. ]]
          Step3:
              按下列公式计算输出 
.. math::
  
  Output[i] = C\_ * Theta[i]^T

参数：
	- **theta** (Variable)： 一类具有形状为[N, 2, 3]的仿射变换参数
	- **out_shape** (Variable | list | tuple)：具有格式[N, C, H, W]的目标输出的shape，out_shape可以是变量、列表或元组。
	- **name** (str|None): 此层的名称(可选)。如果没有设置，将自动命名。

返回： Variable: 形为[N, H, W, 2]的输出。

抛出异常： ValueError: 如果输入了不支持的参数类型

**代码示例：**

.. code-block:: python

    theta = fluid.layers.data(name="x", shape=[2, 3], dtype="float32")
    out_shape = fluid.layers.data(name="y", shape=[-1], dtype="float32")
    data = fluid.layers.affine_grid(theta, out_shape)
    # or
    data = fluid.layers.affine_grid(theta, [5, 3, 28, 28])









.. _cn_api_fluid_layers_autoincreased_step_counter:

autoincreased_step_counter
-------------------------------

.. py:function:: paddle.fluid.layers.autoincreased_step_counter(counter_name=None, begin=1, step=1)

创建一个自增变量，每个mini-batch返回主函数运行次数，变量自动加1，默认初始值为1.

参数：
    - **counter_name** (str)-计数名称，默认为 ``@STEP_COUNTER@``
    - **begin** (int)-开始计数
    - **step** (int)-执行之间增加的步数

返回：全局运行步数

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    global_step = fluid.layers.autoincreased_step_counter(
        counter_name='@LR_DECAY_COUNTER@', begin=begin, step=1)









.. _cn_api_fluid_layers_batch_norm:

batch_norm
-------------------------------

.. py:function:: paddle.fluid.layers.batch_norm(input, act=None, is_test=False, momentum=0.9, epsilon=1e-05, param_attr=None, bias_attr=None, data_layout='NCHW', in_place=False, name=None, moving_mean_name=None, moving_variance_name=None, do_model_average_for_mean_and_var=False, fuse_with_relu=False)

批正则化层（Batch Normalization Layer）

可用作conv2d和全链接操作的正则化函数。该层需要的数据格式如下：

1.NHWC[batch,in_height,in_width,in_channels]
2.NCHW[batch,in_channels,in_height,in_width]

更多详情请参考 : `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift <https://arxiv.org/pdf/1502.03167.pdf>`_

``input`` 是mini-batch的输入特征。

.. math::
    \mu_{\beta}        &\gets \frac{1}{m} \sum_{i=1}^{m} x_i                                 \quad &// mini-batch-mean \\
    \sigma_{\beta}^{2} &\gets \frac{1}{m} \sum_{i=1}^{m}(x_i - \mu_{\beta})^2               \quad &// mini-batch-variance \\
    \hat{x_i}          &\gets \frac{x_i - \mu_\beta} {\sqrt{\sigma_{\beta}^{2} + \epsilon}}  \quad &// normalize \\
    y_i &\gets \gamma \hat{x_i} + \beta                                                      \quad &// scale-and-shift

参数：
    - **input** (Variable) - 输入变量，为LoDTensor
    - **act** （string，默认None）- 激活函数类型，linear|relu|prelu|...
    - **is_test** （bool,默认False） - 用于训练
    - **momentum** （float，默认0.9）- （暂无说明，待更新）
    - **epsilon** （float，默认1e-05）- （暂无说明，待更新）
    - **param_attr** （ParamAttr|None） - batch_norm参数范围的属性，如果设为None或者是ParamAttr的一个属性，batch_norm创建ParamAttr为param_attr。如果没有设置param_attr的初始化函数，参数初始化为Xavier。默认：None
    - **bias_attr** （ParamAttr|None） - batch_norm bias参数的属性，如果设为None或者是ParamAttr的一个属性，batch_norm创建ParamAttr为bias_attr。如果没有设置bias_attr的初始化函数，参数初始化为0。默认：None
    - **data_layout** （string,默认NCHW) - NCHW|NHWC
    - **in_place** （bool，默认False）- 得出batch norm可复用记忆的输入和输出
    - **name** （string，默认None）- 该层名称（可选）。若设为None，则自动为该层命名
    - **moving_mean_name** （string，默认None）- moving_mean的名称，存储全局Mean
    - **moving_variance_name** （string，默认None）- moving_variance的名称，存储全局变量
    - **do_model_average_for_mean_and_var** （bool，默认False）- 是否为mean和variance做模型均值
    - **fuse_with_relo** （bool）- 如果为True，batch norm后该操作符执行relu

返回： 张量，在输入中运用批正则后的结果

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    hidden1 = fluid.layers.fc(input=x, size=200, param_attr='fc1.w')
    hidden2 = fluid.layers.batch_norm(input=hidden1)











.. _cn_api_fluid_layers_beam_search:

beam_search
-------------------------------

.. py:function:: paddle.fluid.layers.beam_search(pre_ids, pre_scores, ids, scores, beam_size, end_id, level=0, name=None)

在机器翻译任务中，束搜索(Beam search)是选择候选词的一种经典算法

更多细节参考 `Beam Search <https://en.wikipedia.org/wiki/Beam_search>`_

该层对束在一时间步中进行搜索。根据候选词使用于源句子所得的 ``scores`` , 从候选词 ``ids`` 中选择当前步骤的 top-K （最佳K）候选词的id，其中 ``K`` 是 ``beam_size`` ， ``ids`` ， ``scores`` 是计算单元的预测结果。 另外， ``pre_id`` 和 ``pre_scores`` 是上一步中 ``beam_search`` 的输出，用于特殊处理结束边界。

注意，传入的 ``scores`` 应该是累积分数，并且，在计算累积分数之前应该使用额外的 operators 进行长度惩罚，也建议在计算前查找top-K，然后使用top-K候选项。

有关完全波束搜索用法演示，请参阅以下示例：
  
     fluid/tests/book/test_machine_translation.py
  


参数:
  - **pre_ids** （Variable） -  LodTensor变量，它是上一步 ``beam_search`` 的输出。在第一步中。它应该是LodTensor，shape为 :math:`(batch\_size，1)` ， :math:`lod [[0,1，...，batch\_size]，[0,1，...，batch\_size]]` 
  - **pre_scores** （Variable） -  LodTensor变量，它是上一步中beam_search的输出
  - **ids** （Variable） - 包含候选ID的LodTensor变量。shpae为 :math:`（batch\_size×beam\_ize，K）` ，其中 ``K`` 应该是 ``beam_size``
  - **scores** （Variable） - 与 ``ids`` 及其shape对应的累积分数的LodTensor变量, 与 ``ids`` 的shape相同。
  - **beam_size** （int） - 束搜索中的束宽度。
  - **end_id** （int） - 结束标记的id。
  - **level** （int，default 0） - **可忽略，当前不能更改** 。它表示lod的源级别，解释如下。 ``ids`` 的 lod 级别应为2.第一级是源级别， 描述每个源句子（beam）的前缀（分支）的数量，第二级是描述这些候选者属于前缀的句子级别的方式。链接前缀和所选候选者的路径信息保存在lod中。
  - **name** （str | None） - 该层的名称（可选）。如果设置为None，则自动命名该层。

返回：LodTensor pair ， 包含所选的ID和相应的分数

返回类型：Variable

**代码示例**

..  code-block:: python
    
    # 假设 `probs` 包含计算神经元所得的预测结果
    # `pre_ids` 和 `pre_scores` 为beam_search之前时间步的输出
    topk_scores, topk_indices = layers.topk(probs, k=beam_size)
    accu_scores = layers.elementwise_add(
                                          x=layers.log(x=topk_scores)),
                                          y=layers.reshape(
                                              pre_scores, shape=[-1]),
                                          axis=0)
    selected_ids, selected_scores = layers.beam_search(
                                          pre_ids=pre_ids,
                                          pre_scores=pre_scores,
                                          ids=topk_indices,
                                          scores=accu_scores,
                                          beam_size=beam_size,
                                          end_id=end_id)











.. _cn_api_fluid_layers_beam_search_decode:

beam_search_decode
-------------------------------

.. py:function:: paddle.fluid.layers.beam_search_decode(ids, scores, beam_size, end_id, name=None)

束搜索层（Beam Search Decode Layer）通过回溯LoDTensorArray ids，为每个源语句构建完整假设，LoDTensorArray ``ids`` 的lod可用于恢复束搜索树中的路径。请参阅下面的demo中的束搜索使用示例：

    ::

        fluid/tests/book/test_machine_translation.py

参数:
        - **id** (Variable) - LodTensorArray，包含所有回溯步骤重中所需的ids。
        - **score** (Variable) - LodTensorArra，包含所有回溯步骤对应的score。
        - **beam_size** (int) - 束搜索中波束的宽度。
        - **end_id** (int) - 结束token的id。
        - **name** (str|None) - 该层的名称(可选)。如果设置为None，该层将被自动命名。
    
返回：	LodTensor 对（pair）， 由生成的id序列和相应的score序列组成。两个LodTensor的shape和lod是相同的。lod的level=2，这两个level分别表示每个源句有多少个假设，每个假设有多少个id。

返回类型:	变量（variable）


**代码示例**

.. code-block:: python
            
	    # 假设 `ids` 和 `scores` 为 LodTensorArray变量，它们保留了
            # 选择出的所有时间步的id和score
            finished_ids, finished_scores = layers.beam_search_decode(
                ids, scores, beam_size=5, end_id=0)









.. _cn_api_fluid_layers_bilinear_tensor_product:

bilinear_tensor_product
-------------------------------

.. py:function:: paddle.fluid.layers.bilinear_tensor_product(x, y, size, act=None, name=None, param_attr=None, bias_attr=None)

该层对两个输入执行双线性张量积。

例如:

.. math::
       out{i} = x * W_{i} * {y^\mathrm{T}}, i=0,1,...,size-1

在这个公式中：
	- :math:`x`: 第一个输入，包含M个元素，形状为[batch_size, M]
	- :math:`y`: 第二个输入，包含N个元素，形状为[batch_size, N]
	- :math:`W_{i}`: 第i个被学习的权重，形状是[M, N]
	- :math:`out_{i}`: out的第i个元素，形状是[batch_size, size]
	- :math:`y^\mathrm{T}`: :math:`y_{2}` 的转置

参数：
    - **x** (Variable): 2-D 输入张量，形状为 [batch_size, M]
    - **y** (Variable): 2-D 输入张量，形状为 [batch_size, N]
    - **size** (int): 此层的维度，
    - **act** (str, default None): 应用到该层输出的激活函数
    - **name** (str, default None): 该层的名称
    - **param_attr** (ParamAttr, default None): 可学习参数/权重（w） 的参数属性
    - **bias_attr** (ParamAttr, default None): 偏差的参数属性，如果设置为False，则不会向输出单元添加偏差。如果设置为零，偏差初始化为零。默认值:None

返回： Variable: 一个形为[batch_size, size]的2-D张量

**代码示例：**

.. code-block:: python

	tensor = bilinear_tensor_product(x=layer1, y=layer2, size=1000)










.. _cn_api_fluid_layers_brelu:

brelu
-------------------------------

.. py:function:: paddle.fluid.layers.brelu(x, t_min=0.0, t_max=24.0, name=None)


BRelu 激活函数

.. math::   out=max(min(x,tmin),tmax)

参数: 
    - **x** (Variable) - BReluoperator的输入
    - **t_min** (FLOAT|0.0) - BRelu的最小值
    - **t_max** (FLOAT|24.0) - BRelu的最大值
    - **name** (str|None) - 该层的名称(可选)。如果设置为None，该层将被自动命名


**代码示例：**

.. code-block:: python

    x = fluid.layers.data(name="x", shape=[2,3,16,16], dtype=”float32”)
    y = fluid.layers.brelu(x, t_min=1.0, t_max=20.0)






.. _cn_api_fluid_layers_chunk_eval:

chunk_eval
-------------------------------

.. py:function:: paddle.fluid.layers.chunk_eval(input, label, chunk_scheme, num_chunk_types, excluded_chunk_types=None)

块估计（Chunk Evaluator）

该功能计算并输出块检测（chunk detection）的准确率、召回率和F1值。

chunking的一些基础请参考 `Chunking with Support Vector Machines <https://aclanthology.info/pdf/N/N01/N01-1025.pdf>`_

ChunkEvalOp计算块检测（chunk detection）的准确率、召回率和F1值，并支持IOB，IOE，IOBES和IO标注方案。以下是这些标注方案的命名实体（NER）标注例子：

::


    ====== ====== ======  =====  ==  ============   =====  ===== =====  ==  =========
           Li     Ming    works  at  Agricultural   Bank   of    China  in  Beijing.
    ====== ====== ======  =====  ==  ============   =====  ===== =====  ==  =========
    IO     I-PER  I-PER   O      O   I-ORG          I-ORG  I-ORG I-ORG  O   I-LOC
    IOB    B-PER  I-PER   O      O   B-ORG          I-ORG  I-ORG I-ORG  O   B-LOC
    IOE    I-PER  E-PER   O      O   I-ORG          I-ORG  I-ORG E-ORG  O   E-LOC
    IOBES  B-PER  E-PER   O      O   I-ORG          I-ORG  I-ORG E-ORG  O   S-LOC
    ====== ====== ======  =====  ==  ============   =====  ===== =====  ==  =========

有三种块类别（命名实体类型），包括PER（人名），ORG（机构名）和LOC（地名），标签形式为标注类型（tag type）-块类型（chunk type）。

由于计算实际上用的是标签id而不是标签，需要额外注意将标签映射到相应的id，这样CheckEvalOp才可运行。关键在于id必须在列出的等式中有效。

::


    tag_type = label % num_tag_type
    chunk_type = label / num_tag_type

num_tag_type是标注规则中的标签类型数，num_chunk_type是块类型数，tag_type从下面的表格中获取值。

::


    Scheme Begin Inside End   Single
    plain   0     -      -     -
    IOB     0     1      -     -
    IOE     -     0      1     -
    IOBES   0     1      2     3

仍以NER为例，假设标注规则是IOB块类型为ORG，PER和LOC。为了满足以上等式，标签图如下：

::


    B-ORG  0
    I-ORG  1
    B-PER  2
    I-PER  3
    B-LOC  4
    I-LOC  5
    O      6

不难证明等式的块类型数为3，IOB规则中的标签类型数为2.例如I-LOC的标签id为5，I-LOC的标签类型id为1，I-LOC的块类型id为2，与等式的结果一致。

参数：
    - **input** (Variable) - 网络的输出预测
    - **label** (Variable) - 测试数据集的标签
    - **chunk_scheme** (str) - 标注规则，表示如何解码块。必须数IOB，IOE，IOBES或者plain。详情见描述
    - **num_chunk_types** (int) - 块类型数。详情见描述
    - **excluded_chunk_types** (list) - 列表包含块类型id，表示不在计数内的块类型。详情见描述

返回：元组（tuple），包含precision, recall, f1_score, num_infer_chunks, num_label_chunks, num_correct_chunks

返回类型：tuple（元组）

**代码示例**：

.. code-block:: python:

    crf = fluid.layers.linear_chain_crf(
        input=hidden, label=label, param_attr=ParamAttr(name="crfw"))
    crf_decode = fluid.layers.crf_decoding(
        input=hidden, param_attr=ParamAttr(name="crfw"))
    fluid.layers.chunk_eval(
        input=crf_decode,
        label=label,
        chunk_scheme="IOB",
        num_chunk_types=(label_dict_len - 1) / 2)









.. _cn_api_fluid_layers_clip:

clip
-------------------------------

.. py:function:: paddle.fluid.layers.clip(x, min, max, name=None)
        
clip算子

clip运算符限制给定输入的值在一个区间内。间隔使用参数"min"和"max"来指定：公式为

.. math:: 
        Out=min(max(X,min),max)

参数：
        - **x** （Variable）- （Tensor）clip运算的输入，维数必须在[1,9]之间。
        - **min** （FLOAT）- （float）最小值，小于该值的元素由min代替。
        - **max** （FLOAT）- （float）最大值，大于该值的元素由max替换。
        - **name** （basestring | None）- 输出的名称。

返回：        （Tensor）clip操作后的输出和输入（X）具有形状（shape）

返回类型：        输出（Variable）。        

**代码示例：**

.. code-block:: python

    input = fluid.layers.data(
        name='data', shape=[1], dtype='float32')
    reward = fluid.layers.clip(x=input, min=-1.0, max=1.0)






.. _cn_api_fluid_layers_clip_by_norm:

clip_by_norm
-------------------------------

.. py:function:: paddle.fluid.layers.clip_by_norm(x, max_norm, name=None)
     
ClipByNorm算子

此运算符将输入 ``X`` 的L2范数限制在 ``max_norm`` 内。如果 ``X`` 的L2范数小于或等于 ``max_norm``  ，则输出（Out）将与 ``X`` 相同。如果X的L2范数大于 ``max_norm`` ，则 ``X`` 将被线性缩放，使得输出（Out）的L2范数等于 ``max_norm`` ，如下面的公式所示：

.. math:: 
         Out = \frac{max\_norm * X}{norm(X)} 

其中， :math:`norm（X）` 代表 ``x`` 的L2范数。


参数：
        - **x** (Variable)- (Tensor) clip_by_norm运算的输入，维数必须在[1,9]之间。
        - **max_norm** (float)- 最大范数值。
        - **name** (basestring | None)- 输出的名称。

返回：        (Tensor)clip_by_norm操作后的输出和输入(X)具有形状(shape).

返回类型：       Variable        

**代码示例：**

.. code-block:: python

    input = fluid.layers.data(
        name='data', shape=[1], dtype='float32')
    reward = fluid.layers.clip_by_norm(x=input, max_norm=1.0)








.. _cn_api_fluid_layers_conv2d:

conv2d
-------------------------------

.. py:function:: paddle.fluid.layers.conv2d(input, num_filters, filter_size, stride=1, padding=0, dilation=1, groups=None, param_attr=None, bias_attr=None, use_cudnn=True, act=None, name=None)

卷积二维层（convolution2D layer）根据输入、滤波器（filter）、步长（stride）、填充（padding）、dilations、一组参数计算输出。输入和输出是NCHW格式，N是批尺寸，C是通道数，H是特征高度，W是特征宽度。滤波器是MCHW格式，M是输出图像通道数，C是输入图像通道数，H是滤波器高度，W是滤波器宽度。如果组数大于1，C等于输入图像通道数除以组数的结果。详情请参考UFLDL's : `卷积 <http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/>`_ 。如果提供了bias属性和激活函数类型，bias会添加到卷积（convolution）的结果中相应的激活函数会作用在最终结果上。

对每个输入X，有等式：

.. math::

    Out = \sigma \left ( W * X + b \right )

其中：
    - :math:`X` ：输入值，NCHW格式的张量（Tensor）
    - :math:`W` ：滤波器值，MCHW格式的张量（Tensor）
    - :math:`*` ： 卷积操作
    - :math:`b` ：Bias值，二维张量（Tensor），shape为[M,1]
    - :math:`\sigma` ：激活函数
    - :math:`Out` ：输出值，*Out*和**X**的shape可能不同

**示例**

- 输入：

  输入shape：:math:`( N,C_{in},H_{in},W_{in} )`

  滤波器shape： :math:`( C_{out},C_{in},H_{f},W_{f} )`

- 输出：

  输出shape： :math:`( N,C_{out},H_{out},W_{out} )`

其中

.. math::

    H_{out} = \frac{\left ( H_{in}+2*paddings[0]-\left ( dilations[0]*\left ( H_{f}-1 \right )+1 \right ) \right )}{strides[0]}+1

    W_{out} = \frac{\left ( H_{in}+2*paddings[1]-\left ( dilations[1]*\left ( W_{f}-1 \right )+1 \right ) \right )}{strides[1]}+1

参数：
    - **input** (Variable) - 格式为[N,C,H,W]格式的输入图像
    - **num_fliters** (int) - 滤波器数。和输出图像通道相同
    - **filter_size** (int|tuple|None) - 滤波器大小。如果filter_size是一个元组，则必须包含两个整型数，（filter_size，filter_size_W）。否则，滤波器为square
    - **stride** (int|tuple) - 步长(stride)大小。如果步长（stride）为元组，则必须包含两个整型数，（stride_H,stride_W）。否则，stride_H = stride_W = stride。默认：stride = 1
    - **padding** (int|tuple) - 填充（padding）大小。如果填充（padding）为元组，则必须包含两个整型数，（padding_H,padding_W)。否则，padding_H = padding_W = padding。默认：padding = 0
    - **dilation** (int|tuple) - 膨胀（dilation）大小。如果膨胀（dialation）为元组，则必须包含两个整型数，（dilation_H,dilation_W）。否则，dilation_H = dilation_W = dilation。默认：dilation = 1
    - **groups** (int) - 卷积二维层（Conv2D Layer）的组数。根据Alex Krizhevsky的深度卷积神经网络（CNN）论文中的成组卷积：当group=2，滤波器的前一半仅和输入通道的前一半连接。滤波器的后一半仅和输入通道的后一半连接。默认：groups = 1
    - **param_attr** (ParamAttr|None) - conv2d的可学习参数/权重的参数属性。如果设为None或者ParamAttr的一个属性，conv2d创建ParamAttr为param_attr。如果param_attr的初始化函数未设置，参数则初始化为 :math:`Normal(0.0,std)` ，并且std为 :math:`\frac{2.0}{filter\_elem\_num}^{0.5}` 。默认为None
    - **bias_attr** (ParamAttr|bool|None) - conv2d bias的参数属性。如果设为False，则没有bias加到输出。如果设为None或者ParamAttr的一个属性，conv2d创建ParamAttr为bias_attr。如果bias_attr的初始化函数未设置，bias初始化为0.默认为None
    - **use_cudnn** （bool） - 是否用cudnn核，仅当下载cudnn库才有效。默认：True
    - **act** (str) - 激活函数类型，如果设为None，则未添加激活函数。默认：None
    - **name** (str|None) - 该层名称（可选）。若设为None，则自动为该层命名。

返回：张量，存储卷积和非线性激活结果

返回类型：变量（Variable）

抛出异常:
  - ``ValueError`` - 如果输入shape和filter_size，stride,padding和group不匹配。

**代码示例**：

.. code-block:: python

    data = fluid.layers.data(name='data', shape=[3, 32, 32], dtype='float32')
    conv2d = fluid.layers.conv2d(input=data, num_filters=2, filter_size=3, act="relu")











.. _cn_api_fluid_layers_conv2d_transpose:

conv2d_transpose
-------------------------------

.. py:function:: paddle.fluid.layers.conv2d_transpose(input, num_filters, output_size=None, filter_size=None, padding=0, stride=1, dilation=1, groups=None, param_attr=None, bias_attr=None, use_cudnn=True, act=None, name=None)

2-D卷积转置层（Convlution2D transpose layer）

该层根据 输入（input）、滤波器（filter）和卷积核膨胀（dilations）、步长（stride）、填充（padding）来计算输出。输入(Input)和输出(Output)为NCHW格式，其中 ``N`` 为batch大小， ``C`` 为通道数（channel），``H`` 为特征高度， ``W`` 为特征宽度。参数(膨胀、步长、填充)分别都包含两个元素。这两个元素分别表示高度和宽度。欲了解卷积转置层细节，请参考下面的说明和 参考文献_ 。如果参数 ``bias_attr`` 和 ``act`` 不为 ``None``，则在卷积的输出中加入偏置，并对最终结果应用相应的激活函数。

.. _参考文献: http://www.matthewzeiler.com/wp-content/uploads/2017/07/cvpr2010.pdf

输入 :math:`X` 和输出 :math:`Out` 函数关系如下：

.. math::			   
                        Out=\sigma (W*X+b)\\

其中：
    -  :math:`X` : 输入张量，具有 ``NCHW`` 格式

    -  :math:`W` : 滤波器张量，，具有 ``NCHW`` 格式

    -  :math:`*` : 卷积操作

    -  :math:`b` : 偏置（bias），二维张量，shape为 ``[m,1]``

    -  :math:`σ` : 激活函数
 
    -  :math:`Out` : 输出值，Out和 ``X`` 的 ``shape`` 可能不一样

**样例**：

输入：

.. math:: 

    输入张量的shape :  （N，C_{in}， H_{in}， W_{in})

    滤波器（filter）shape ： （C_{in}, C_{out}, H_f, W_f)  

输出：
        
.. math:: 
    输出张量的 shape ： （N，C_{out}, H_{out}, W_{out})

其中

.. math:: 

        & H'_{out} = (H_{in}-1)*strides[0]-2*paddings[0]+dilations[0]*(H_f-1)+1\\
        & W'_{out} = (W_{in}-1)*strides[1]-2*paddings[1]+dilations[1]*(W_f-1)+1 \\
        & H_{out}\in[H'_{out},H'_{out} + strides[0])\\
        & W_{out}\in[W'_{out},W'_{out} + strides[1])\\

            

参数:
	- **input** （Variable）- 输入张量，格式为[N, C, H, W]
	- **num_filters** (int) - 滤波器（卷积核）的个数，与输出的图片的通道数（ channel ）相同
	- **output_size** (int|tuple|None) - 输出图片的大小。如果output_size是一个元组（tuple），则该元形式为（image_H,image_W),这两个值必须为整型。如果output_size=None,则内部会使用filter_size、padding和stride来计算output_size。如果output_size和filter_size是同时指定的，那么它们应满足上面的公式。
	- **filter_size** (int|tuple|None) - 滤波器大小。如果filter_size是一个tuple，则形式为(filter_size_H, filter_size_W)。否则，滤波器将是一个方阵。如果filter_size=None，则内部会计算输出大小。
	- **padding** (int|tuple) - 填充大小。如果padding是一个元组，它必须包含两个整数(padding_H、padding_W)。否则，padding_H = padding_W = padding。默认:padding = 0。
	- **stride** (int|tuple) - 步长大小。如果stride是一个元组，那么元组的形式为(stride_H、stride_W)。否则，stride_H = stride_W = stride。默认:stride = 1。
	- **dilation** (int|元组) - 膨胀大小。如果dilation是一个元组，那么元组的形式为(dilation_H, dilation_W)。否则，dilation_H = dilation_W = dilation_W。默认:dilation= 1。
	- **groups** (int) - Conv2d转置层的groups个数。从Alex Krizhevsky的CNN Deep论文中的群卷积中受到启发，当group=2时，前半部分滤波器只连接到输入通道的前半部分，而后半部分滤波器只连接到输入通道的后半部分。默认值:group = 1。
	- **param_attr** (ParamAttr|None) - conv2d_transfer中可学习参数/权重的属性。如果param_attr值为None或ParamAttr的一个属性，conv2d_transfer使用ParamAttrs作为param_attr的值。如果没有设置的param_attr初始化器，那么使用Xavier初始化。默认值:None。
	- **bias_attr** (ParamAttr|bool|None) - conv2d_tran_bias中的bias属性。如果设置为False，则不会向输出单元添加偏置。如果param_attr值为None或ParamAttr的一个属性，将conv2d_transfer使用ParamAttrs作为，bias_attr。如果没有设置bias_attr的初始化器，bias将初始化为零。默认值:None。
	- **use_cudnn** (bool) - 是否使用cudnn内核，只有已安装cudnn库时才有效。默认值:True。
	- **act** (str) -  激活函数类型，如果设置为None，则不使用激活函数。默认值:None。
	-  **name** (str|None) -  该layer的名称(可选)。如果设置为None， 将自动命名该layer。默认值:True。


返回：	存储卷积转置结果的张量。

返回类型:	变量（variable）

抛出异常:
    -  ``ValueError`` : 如果输入的shape、filter_size、stride、padding和groups不匹配，抛出ValueError

**代码示例**

..  code-block:: python
  
    data = fluid.layers.data(name='data', shape=[3, 32, 32], dtype='float32')
    conv2d_transpose = fluid.layers.conv2d_transpose(input=data, num_filters=2, filter_size=3)
    
  









.. _cn_api_fluid_layers_conv3d:

conv3d
-------------------------------

.. py:function:: paddle.fluid.layers.conv3d(input, num_filters, filter_size, stride=1, padding=0, dilation=1, groups=None, param_attr=None, bias_attr=None, use_cudnn=True, act=None, name=None)

卷积三维层（convolution3D layer）根据输入、滤波器（filter）、步长（stride）、填充（padding）、膨胀（dilations）、组数参数计算得到输出。输入和输出是NCHW格式，N是批尺寸，C是通道数，H是特征高度，W是特征宽度。卷积三维（Convlution3D）和卷积二维（Convlution2D）相似，但多了一维深度（depth）。如果提供了bias属性和激活函数类型，bias会添加到卷积（convolution）的结果中相应的激活函数会作用在最终结果上。

对每个输入X，有等式：

.. math::


    Out = \sigma \left ( W * X + b \right )

其中：
    - :math:`X` ：输入值，NCHW格式的张量（Tensor）
    - :math:`W` ：滤波器值，MCHW格式的张量（Tensor）
    - :math:`*` ： 卷积操作
    - :math:`b` ：Bias值，二维张量（Tensor），形为[M,1]
    - :math:`\sigma` ：激活函数
    - :math:`Out` ：输出值, 和 :math:`X` 的形状可能不同

**示例**

- 输入：
    输入shape： :math:`( N,C_{in},H_{in},W_{in}  )` 

    滤波器shape： :math:`( C_{out},C_{in},H_{f},W_{f} )` 
- 输出：
    输出shape： :math:`( N,C_{out},H_{out},W_{out} )` 

其中

.. math::


    D_{out} = \frac{\left ( D_{in}+2*paddings[0]-\left ( dilations[0]*\left ( D_{f}-1 \right )+1 \right ) \right )}{strides[0]}+1

    H_{out} = \frac{\left ( H_{in}+2*paddings[1]-\left ( dilations[1]*\left ( H_{f}-1 \right )+1 \right ) \right )}{strides[1]}+1

    W_{out} = \frac{\left ( W_{in}+2*paddings[2]-\left ( dilations[2]*\left ( W_{f}-1 \right )+1 \right ) \right )}{strides[2]}+1

参数：
    - **input** (Variable) - 格式为[N,C,H,W]格式的输入图像
    - **num_fliters** (int) - 滤波器数。和输出图像通道相同
    - **filter_size** (int|tuple|None) - 滤波器大小。如果filter_size是一个元组，则必须包含两个整型数，（filter_size，filter_size_W）。否则，滤波器为square
    - **stride** (int|tuple) - 步长(stride)大小。如果步长（stride）为元组，则必须包含两个整型数，（stride_H,stride_W）。否则，stride_H = stride_W = stride。默认：stride = 1
    - **padding** (int|tuple) - 填充（padding）大小。如果填充（padding）为元组，则必须包含两个整型数，（padding_H,padding_W)。否则，padding_H = padding_W = padding。默认：padding = 0
    - **dilation** (int|tuple) - 膨胀（dilation）大小。如果膨胀（dialation）为元组，则必须包含两个整型数，（dilation_H,dilation_W）。否则，dilation_H = dilation_W = dilation。默认：dilation = 1
    - **groups** (int) - 卷积二维层（Conv2D Layer）的组数。根据Alex Krizhevsky的深度卷积神经网络（CNN）论文中的成组卷积：当group=2，滤波器的前一半仅和输入通道的前一半连接。滤波器的后一半仅和输入通道的后一半连接。默认：groups = 1
    - **param_attr** (ParamAttr|None) - conv2d的可学习参数/权重的参数属性。如果设为None或者ParamAttr的一个属性，conv2d创建ParamAttr为param_attr。如果param_attr的初始化函数未设置，参数则初始化为 :math:`Normal(0.0,std)`，并且std为 :math:`\left ( \frac{2.0}{filter\_elem\_num} \right )^{0.5}` 。默认为None
    - **bias_attr** (ParamAttr|bool|None) - conv2d bias的参数属性。如果设为False，则没有bias加到输出。如果设为None或者ParamAttr的一个属性，conv2d创建ParamAttr为bias_attr。如果bias_attr的初始化函数未设置，bias初始化为0.默认为None
    - **use_cudnn** （bool） - 是否用cudnn核，仅当下载cudnn库才有效。默认：True
    - **act** (str) - 激活函数类型，如果设为None，则未添加激活函数。默认：None
    - **name** (str|None) - 该层名称（可选）。若设为None，则自动为该层命名。

返回：张量，存储卷积和非线性激活结果

返回类型：变量（Variable）

抛出异常：
  - ``ValueError`` - 如果 ``input`` 的形和 ``filter_size`` ， ``stride`` , ``padding`` 和 ``group`` 不匹配。

**代码示例**：

.. code-block:: python

    data = fluid.layers.data(name='data', shape=[3, 12, 32, 32], dtype='float32')
    conv3d = fluid.layers.conv3d(input=data, num_filters=2, filter_size=3, act="relu")









.. _cn_api_fluid_layers_conv3d_transpose:

conv3d_transpose
-------------------------------

.. py:function:: paddle.fluid.layers.conv3d_transpose(input, num_filters, output_size=None, filter_size=None, padding=0, stride=1, dilation=1, groups=None, param_attr=None, bias_attr=None, use_cudnn=True, act=None, name=None)

3-D卷积转置层（Convlution3D transpose layer)

该层根据 输入（input）、滤波器（filter）和卷积核膨胀（dilations）、步长（stride）、填充来计算输出。输入(Input)和输出(Output)为NCDHW格式。其中 ``N`` 为batch大小， ``C`` 为通道数（channel）, ``D``  为特征深度, ``H`` 为特征高度， ``W`` 为特征宽度。参数(膨胀、步长、填充)分别包含两个元素。这两个元素分别表示高度和宽度。欲了解卷积转置层细节，请参考下面的说明和 参考文献_ 。如果参数 ``bias_attr`` 和 ``act`` 不为None，则在卷积的输出中加入偏置，并对最终结果应用相应的激活函数

.. _参考文献: http://www.matthewzeiler.com/wp-content/uploads/2017/07/cvpr2010.pdf

输入X和输出Out函数关系X，有等式如下：

.. math::
                        \\Out=\sigma (W*X+b)\\

其中：
    -  :math:`X` : 输入张量，具有 ``NCDHW`` 格式

    -  :math:`W` : 滤波器张量，，具有 ``NCDHW`` 格式

    -  :math:`*` : 卷积操作

    -  :math:`b` : 偏置（bias），二维张量，shape为[m,1]

    -  :math:`σ` : 激活函数
 
    -  :math:`Out` : 输出值， ``Out`` 和 ``X`` 的 shape可能不一样


**样例**

输入:

.. math::   
	
		Input shape: (N,C_{in},D_{in},H_{in},W_{in})

		Filter shape: (C_{in},C_{out},D_f,H_f,W_f)

	

输出:

.. math::   
	
		Output shape: (N,C_{out},D_{out},H_{out},W_{out})

	
其中：

.. math::   
		


		D_{out}=(D_{in}-1)*strides[0]-2*paddings[0]+dilations[0]*(D_f-1)+1
	
		H_{out}=(H_{in}-1)*strides[1]-2*paddings[1]+dilations[1]*(H_f-1)+1
	
		W_{out}=(W_{in}-1)*strides[2]-2*paddings[2]+dilations[2]*(W_f-1)+1
		


参数:
	- **input** （Variable）- 输入张量，格式为[N, C, D, H, W]
	- **num_filters** (int) - 滤波器（卷积核）的个数，与输出的图片的通道数（channel）相同
	- **output_size** (int|tuple|None) - 输出图片的大小。如果 ``output_size`` 是一个元组（tuple），则该元形式为（image_H,image_W),这两个值必须为整型。如果 ``output_size=None`` ,则内部会使用filter_size、padding和stride来计算output_size。如果 ``output_size`` 和 ``filter_size`` 是同时指定的，那么它们应满足上面的公式。
	- **filter_size** (int|tuple|None) - 滤波器大小。如果 ``filter_size`` 是一个tuple，则形式为(filter_size_H, filter_size_W)。否则，滤波器将是一个方阵。如果 ``filter_size=None`` ，则内部会计算输出大小。
	- **padding** (int|tuple) - 填充大小。如果 ``padding`` 是一个元组，它必须包含两个整数(padding_H、padding_W)。否则，padding_H = padding_W = padding。默认:padding = 0。
	- **stride** (int|tuple) - 步长大小。如果 ``stride`` 是一个元组，那么元组的形式为(stride_H、stride_W)。否则，stride_H = stride_W = stride。默认:stride = 1。
	- **dilation** (int|元组) - 膨胀大小。如果 ``dilation`` 是一个元组，那么元组的形式为(dilation_H, dilation_W)。否则，dilation_H = dilation_W = dilation_W。默认:dilation= 1。
	- **groups** (int) - Conv2d转置层的groups个数。从Alex Krizhevsky的CNN Deep论文中的群卷积中受到启发，当group=2时，前半部分滤波器只连接到输入通道的前半部分，而后半部分滤波器只连接到输入通道的后半部分。默认值:group = 1。
	- **param_attr** (ParamAttr|None) - conv2d_transfer中可学习参数/权重的属性。如果param_attr值为None或ParamAttr的一个属性，conv2d_transfer使用ParamAttrs作为param_attr的值。如果没有设置的param_attr初始化器，那么使用Xavier初始化。默认值:None。
	- **bias_attr** (ParamAttr|bool|None) - conv2d_tran_bias中的bias属性。如果设置为False，则不会向输出单元添加偏置。如果param_attr值为None或ParamAttr的一个属性，将conv2d_transfer使用ParamAttrs作为，bias_attr。如果没有设置bias_attr的初始化器，bias将初始化为零。默认值:None。
	- **use_cudnn** (bool) - 是否使用cudnn内核，只有已安装cudnn库时才有效。默认值:True。
	- **act** (str) -  激活函数类型，如果设置为None，则不使用激活函数。默认值:None。
	- **name** (str|None) - 该layer的名称(可选)。如果设置为None， 将自动命名该layer。默认值:True。


返回：	存储卷积转置结果的张量。

返回类型:	变量（variable）

抛出异常:
    -  ``ValueError``  : 如果输入的shape、filter_size、stride、padding和groups不匹配，抛出ValueError


**代码示例**

..  code-block:: python
  
    data = fluid.layers.data(name='data', shape=[3, 12, 32, 32], dtype='float32')
    conv3d_transpose = fluid.layers.conv3d_transpose(input=data, num_filters=2, filter_size=3)










.. _cn_api_fluid_layers_cos_sim:

cos_sim 
-------------------------------

.. py:function:: paddle.fluid.layers.cos_sim(X, Y)

余弦相似度运算符（Cosine Similarity Operator）

.. math::

        Out = \frac{X^{T}*Y}{\sqrt{X^{T}*X}*\sqrt{Y^{T}*Y}}

输入X和Y必须具有相同的shape，除非输入Y的第一维为1(不同于输入X)，在计算它们的余弦相似度之前，Y的第一维会被broadcasted，以匹配输入X的shape。

输入X和Y都携带或者都不携带LoD(Level of Detail)信息。但输出仅采用输入X的LoD信息。

参数：
    - **X** (Variable) - cos_sim操作函数的一个输入
    - **Y** (Variable) - cos_sim操作函数的第二个输入

返回：cosine(X,Y)的输出

返回类型：变量（Variable)









.. _cn_api_fluid_layers_crf_decoding:

crf_decoding
-------------------------------

.. py:function::  paddle.fluid.layers.crf_decoding(input, param_attr, label=None)

该函数读取由 ``linear_chain_crf`` 学习的emission feature weights（发射状态特征的权重）和 transition feature weights(转移特征的权重)。
本函数实现了Viterbi算法，可以动态地寻找隐藏状态最可能的序列，该序列也被称为Viterbi路径（Viterbi path），从而得出的标注(tags)序列。

这个运算的结果会随着 ``Label`` 参数的有无而改变：
      
      1. ``Label`` 非None的情况，在实际训练中时常发生。此时本函数会协同 ``chunk_eval`` 工作。本函数会返回一行形为[N X 1]的向量，其中值为0的部分代表该label不适合作为对应结点的标注，值为1的部分则反之。此类型的输出可以直接作为 ``chunk_eval`` 算子的输入
      
      2. 当没有 ``Label`` 时，该函数会执行标准decoding过程

（没有 ``Label`` 时）该运算返回一个形为 [N X 1]的向量，其中元素取值范围为 0 ~ 最大标注个数-1，分别为预测出的标注（tag）所在的索引。
	
参数：	
    - **input** (Variable)(LoDTensor，默认类型为 LoDTensor<float>) — 一个形为 [N x D] 的LoDTensor，其中 N 是mini-batch的大小，D是标注（tag) 的总数。 该输入是 ``linear_chain_crf`` 的 unscaled emission weight matrix （未标准化的发射权重矩阵）
    - **param_attr** (ParamAttr) — 参与训练的参数的属性
    - **label** (Variable)(LoDTensor，默认类型为 LoDTensor<int64_t>) —  形为[N x 1]的正确标注（ground truth）。 该项可选择传入。 有关该参数的更多信息，请详见上述描述

返回：(LoDTensor, LoDTensor<int64_t>)decoding结果。具体内容根据 ``Label`` 参数是否提供而定。请参照函数介绍来详细了解。

返回类型： Variable


**代码示例**

..  code-block:: python

      crf_decode = layers.crf_decoding(
           input=hidden, param_attr=ParamAttr(name="crfw"))












.. _cn_api_fluid_layers_crop:

crop
-------------------------------

.. py:function:: paddle.fluid.layers.crop(x, shape=None, offsets=None, name=None)

根据偏移量（offsets）和形状（shape），裁剪输入张量。

**样例**：

::

    * Case 1:
        Given
            X = [[0, 1, 2, 0, 0]
                 [0, 3, 4, 0, 0]
                 [0, 0, 0, 0, 0]],
        and
            shape = [2, 2],
            offsets = [0, 1],
        output is:
            Out = [[1, 2],
                   [3, 4]].
    * Case 2:
        Given
            X = [[0, 1, 2, 5, 0]
                 [0, 3, 4, 6, 0]
                 [0, 0, 0, 0, 0]],
        and shape is tensor
            shape = [[0, 0, 0]
                     [0, 0, 0]]
        and
            offsets = [0, 1],

        output is:
            Out = [[1, 2, 5],
                   [3, 4, 6]].

 
参数:
  - **x** (Variable): 输入张量。
  - **shape** (Variable|list/tuple of integer) - 输出张量的形状由参数shape指定，它可以是一个变量/整数的列表/整数元组。如果是张量变量，它的秩必须与x相同。该方式适可用于每次迭代时候需要改变输出形状的情况。如果是整数列表/tupe，则其长度必须与x的秩相同
  - **offsets** (Variable|list/tuple of integer|None) - 指定每个维度上的裁剪的偏移量。它可以是一个Variable，或者一个整数list/tupe。如果是一个tensor variable，它的rank必须与x相同，这种方法适用于每次迭代的偏移量（offset）都可能改变的情况。如果是一个整数list/tupe，则长度必须与x的rank的相同，如果shape=None，则每个维度的偏移量为0。
  - **name** (str|None) - 该层的名称(可选)。如果设置为None，该层将被自动命名。

返回: 裁剪张量。

返回类型: 变量（Variable）

抛出异常: 如果形状不是列表、元组或变量，抛出ValueError


**代码示例**:

..  code-block:: python

    x = fluid.layers.data(name="x", shape=[3, 5], dtype="float32")
    y = fluid.layers.data(name="y", shape=[2, 3], dtype="float32")
    crop = fluid.layers.crop(x, shape=y)


    ## or
    z = fluid.layers.data(name="z", shape=[3, 5], dtype="float32")
    crop = fluid.layers.crop(z, shape=[2, 3])










.. _cn_api_fluid_layers_cross_entropy:

cross_entropy
-------------------------------

.. py:function:: paddle.fluid.layers.cross_entropy(input, label, soft_label=False, ignore_index=-100)

该函数定义了输入和标签之间的cross entropy(交叉熵)层。该函数支持standard cross-entropy computation（标准交叉熵损失计算）
以及soft-label cross-entropy computation（软标签交叉熵损失计算）

  1. One-hot cross-entropy算法
     
     soft_label = False, Label[i, 0] 指明样本i的类别所具的索引:        
                            .. math::
                                     \\Y[i]=-log(X[i,Label[i]])\\
  
  2. Soft-label cross-entropy算法
     
     soft_label = True, Label[i, j] 表明样本i对应类别j的soft label(软标签):        
                            .. math::
                                     \\Y[i]= \sum_{j}-Label[i,j]*log(X[i,j])\\
                                     
     **请确保采用此算法时识别为各软标签的概率总和为1**
  
  3. One-hot cross-entropy with vecterized label（使用向量化标签的One-hot）算法
        
     作为 *2* 的特殊情况，当软类标签内部只有一个非零概率元素，且它的值为1，那么 *2* 算法降级为一种仅有one-hot标签的one-hot交叉熵
  
  



参数：  
    - **input** (Variable|list) – 一个形为[N x D]的二维tensor，其中N是batch大小，D是类别（class）数目。 这是由之前的operator计算出的概率，绝大多数情况下是由softmax operator得出的结果
    - **label** (Variable|list) – 一个二维tensor组成的正确标记的数据集(ground truth)。 当 ``soft_label`` 为False时，label为形为[N x 1]的tensor<int64>。 ``soft_label`` 为True时, label是形为 [N x D]的 tensor<float/double>
    - **soft_label** (bool) – 标志位，指明是否需要把给定的标签列表认定为软标签。默认为False。
    - **ignore_index** (int) – 指定一个被无视的目标值，并且这个值不影响输入梯度变化。仅在 ``soft_label`` 为False时生效。 默认值: -100

返回： 一个形为[N x 1]的二维tensor，承载了交叉熵损失

弹出异常： ``ValueError`` 

                        1. 当 ``input`` 的第一维和 ``label`` 的第一维不相等时，弹出异常
                        2. 当 ``soft_label`` 值为True， 且 ``input`` 的第二维和 ``label`` 的第二维不相等时，弹出异常
                        3. 当 ``soft_label`` 值为False，且 ``label`` 的第二维不是1时，弹出异常
                        


**代码示例**

..  code-block:: python

        predict = fluid.layers.fc(input=net, size=classdim, act='softmax')
        cost = fluid.layers.cross_entropy(input=predict, label=label)













.. _cn_api_fluid_layers_ctc_greedy_decoder:

ctc_greedy_decoder
-------------------------------

.. py:function:: paddle.fluid.layers.ctc_greedy_decoder(input, blank, name=None)

此op用于贪婪策略解码序列，步骤如下:
    1. 获取输入中的每一行的最大值索引，也就是numpy.argmax(input, axis=0)。
    2. 对于step1结果中的每个序列，在两个空格之间合并重复部分（即合并重复的上一步中的到的索引值）并删除所有空格。


简单举一个例子，

::

        已知：

        input.data = [[0.6, 0.1, 0.3, 0.1],
                      [0.3, 0.2, 0.4, 0.1],
                      [0.1, 0.5, 0.1, 0.3],
                      [0.5, 0.1, 0.3, 0.1],

                      [0.5, 0.1, 0.3, 0.1],
                      [0.2, 0.2, 0.2, 0.4],
                      [0.2, 0.2, 0.1, 0.5],
                      [0.5, 0.1, 0.3, 0.1]]

        input.lod = [[4, 4]]

        计算过程：

        1. 将argmax的运算结果应用于输入的第一个序列，即 input.data[0:4] 。
           则得出的结果为[[0], [2], [1], [0]]
        2. 合并重复的索引值部分，删除空格，即为0的值。
           则第一个输入序列对应的输出为：[[2], [1]]

        最后

        output.data = [[2],
                       [1],
                       [3]]

        output.lod = [[2, 1]]


参数:
        - **input** (Variable) — (LoDTensor<float>)，变长序列的概率，它是一个具有LoD信息的二维张量。它的形状是[Lp, num_classes + 1]，其中Lp是所有输入序列长度的和，num_classes是真正的类别。(不包括空白标签)。
        - **blank** (int) — Connectionist Temporal Classification (CTC) loss空白标签索引,  属于半开区间[0,num_classes + 1）。
        - **name** (str) — 此层的名称。可选。
   
返回： CTC贪婪解码结果是一个形为(Lp,1)的二维张量，其中Lp是所有输出序列的长度之和。如果结果中的所有序列都为空，则输出LoDTensor 为[-1]，其中LoD[[]] 形为[1,1]。

返回类型： 变量（Variable）
    

**代码示例**

..  code-block:: python
        
    x = fluid.layers.data(name='x', shape=[8], dtype='float32')

    cost = fluid.layers.ctc_greedy_decoder(input=x, blank=0)











.. _cn_api_fluid_layers_dice_loss:

dice_loss
-------------------------------

.. py:function:: paddle.fluid.layers.dice_loss(input, label, epsilon=1e-05)

dice_loss是比较两批数据相似度，通常用于二值图像分割，即标签为二值。
    
dice_loss定义为:

.. math::       
        dice\_loss &= 1- \frac{2 * intersection\_area}{total\_rea}\\
                   &= \frac{(total\_area−intersection\_area)−intersection\_area}{total\_area}\\
                   &= \frac{union\_area−intersection\_area}{total\_area}           

参数:
    - **input** (Variable) - rank>=2的预测。第一个维度是batch大小，最后一个维度是类编号。
    - **label** （Variable）- 与输入tensor rank相同的正确的标注数据（groud truth）。第一个维度是batch大小，最后一个维度是1。
    - **epsilon** (float) - 将会加到分子和分母上。如果输入和标签都为空，则确保dice为1。默认值:0.00001
    
返回: dice_loss shape为[1]。

返回类型:  dice_loss(Variable)

**代码示例**

..  code-block:: python
        
	predictions = fluid.layers.softmax(x)
    	loss = fluid.layers.dice_loss(input=predictions, label=label, 2)











.. _cn_api_fluid_layers_dropout:

dropout
-------------------------------

.. py:function:: Paddle.fluid.layers.dropout(x,dropout_prob,is_test=False,seed=None,name=None,dropout_implementation='downgrade_in_infer')

dropout操作

丢弃或者保持x的每个元素独立。Dropout是一种正则化技术，通过在训练过程中阻止神经元节点间的联合适应性来减少过拟合。根据给定的丢弃概率dropout操作符随机将一些神经元输出设置为0，其他的仍保持不变。

参数：
    - **x** (Variable)-输入张量
    - **dropout_prob** (float)-设置为0的单元的概率
    - **is_test** (bool)-显示是否进行测试用语的标记
    - **seed** (int)-Python整型，用于创建随机种子。如果该参数设为None，则使用随机种子。注：如果给定一个整型种子，始终丢弃相同的输出单元。训练过程中勿用固定不变的种子。
    - **name** (str|None)-该层名称（可选）。如果设置为None,则自动为该层命名
    - **dropout_implementation** (string) -   [‘downgrade_in_infer’(defauld)|’upscale_in_train’] 
      其中:
      1.downgrade_in_infer(default), 在预测时减小输出结果 

        train: out = input * mask 

        inference: out = input * dropout_prob 

        (mask是一个张量，维度和输入维度相同，值为0或1，值为0的比例即为 ``dropout_prob`` )
        
      2.upscale_in_train, 增加训练时的结果

        train: out = input * mask / ( 1.0 - dropout_prob )

        inference: out = input 

        (make是一个张量，维度和输入维度相同，值为0或1，值为0的比例即为 ``dropout_prob`` ）

dropout操作符可以从程序中移除，程序变得高效。

返回：带有x维的张量

返回类型：变量

**代码示例**：

.. code-block:: python

    x = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
    droped = fluid.layers.dropout(x, dropout_prob=0.5)









.. _cn_api_fluid_layers_dynamic_gru:

dynamic_gru
-------------------------------

.. py:function::  paddle.fluid.layers.dynamic_gru(input, size, param_attr=None, bias_attr=None, is_reverse=False, gate_activation='sigmoid', candidate_activation='tanh', h_0=None)



**实现了Gated Recurrent Unit层。**

详细理论介绍，请参照 `Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling`_。

.. _Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling: https://arxiv.org/abs/1412.3555


公式如下：

.. math:: 
  u_{t}=act_g(W_{ux}x_{t}+W_{uh}h_{t-1}+b_{u})
.. math::
  r_{t}=act_g(W_{rx}x_{t}+W_{rh}h_{t-1}+b_{r})
.. math::
  \widetilde{h_{t}}=act_{c}(W_{cx}x_{t}+W_{ch}(r_{t}\odot h_{t-1})+b_c)
.. math::
  h_t=(1-u_t)\odot h_{t-1}+u_t\odot \widetilde{h_t}

其中， :math:`\odot` 为按元素将向量相乘。 :math:`act_g` 是更新门（update gate）和重置门（reset gate）的激励函数(activation)， 常为 :math:`sigmoid` 函数。 :math:`act_c` 是candidate hidden state(候选隐藏状态)的激励函数，常为 :math:`tanh` 。

注意 :math:`W_{ux}x_{t},W_{rx}x_{t},W_{cx}x_{t}` 这些在 input  :math:`x_t` 上的操作不包括在该运算中。用户可以选择性地在GRU层之前使用FC层来进行这一操作。



参数:
  - **input** (Variable) – dynamic_gru层的输入, 支持variable time length input sequence（可变时长输入序列）。 本变量底层的tensor是一个(T×3D)矩阵， 其中T是该mini-batch中总时间步数， D是隐藏状态的规模（hidden size）。
  - **size** (int) – GRU cell的维度
  - **param_attr** (ParamAttr|None)  –  可学习的隐藏层权重矩阵的参数属性。 
    注意：
                                    - 该矩阵为一个（T X 3D）矩阵。其中D为隐藏状态的规模（hidden size）
                                    - 该矩阵的所有元素由两部分组成。一是update gate和reset gate的权重，形为（D X 2D)，二是候选隐藏状态（candidate hidden state）的权重，形为 (D X D)
    如果该函数参数被设为None或者 ``ParamAttr`` 类的属性之一，则会生成一个 ``ParamAttr`` 类的对象作为param_attr。如果param_attr未被初始化（即其构造函数未被设置），Xavier会负责初始化它。 默认值为None。
  - **bias_attr** (ParamAttr|bool|None) - GRU层bias的参数属性。该（1 X 3D）形的bias变量将会连结（concatenate）在update gate（更新门）、reset gate（重置门）、candidate calculations（候选隐藏状态计算）后。如果值为False，将没有bias会应用到上述三个过程中。如果该函数参数被设为None或者 ``ParamAttr`` 类的属性之一， ``dynamic_gru`` 会生成一个 ``ParamAttr`` 类的对象作为param_attr。如果bias_attr未被初始化（即其构造函数未被设置），则它会被初始化为0。默认值为None。
  - **is_reverse** (bool) –是否计算反GRU(reversed GRU)，默认为False
  - **gate_activation** (str) – update gate 和 reset gate的激励函数（activation）。 可选择[“sigmoid”, “tanh”, “relu”, “identity”]其一, 默认为 “sigmoid”
  - **candidate_activation** (str) – candidate hidden state（候选隐藏状态）计算所需的激励函数（activation）。 可从[“sigmoid”, “tanh”, “relu”, “identity”]中选择, 默认为 “tanh”
  - **h_0** (Variable) – 该函数参数为初始隐藏状态。若未赋值，则默认为0。它是一个 (N x D) tensor, 其中 N 为输入mini-batch的总时间步数， D 为 隐藏状态规模(hidden size)
  
  
返回：	GRU的隐藏状态(hidden state)。形为（T X D），序列长度和输入相同。

返回类型:	变量（variable）


**代码示例**

..  code-block:: python

    dict_dim, emb_dim = 128, 64
    data = fluid.layers.data(name='sequence', shape=[1],
                             dtype='int32', lod_level=1)
    emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])
    hidden_dim = 512
    x = fluid.layers.fc(input=emb, size=hidden_dim * 3)
    hidden = fluid.layers.dynamic_gru(input=x, size=hidden_dim)
















.. _cn_api_fluid_layers_dynamic_lstm:

dynamic_lstm
-------------------------------

.. py:function::  paddle.fluid.layers.dynamic_lstm(input, size, h_0=None, c_0=None, param_attr=None, bias_attr=None, use_peepholes=True, is_reverse=False, gate_activation='sigmoid', cell_activation='tanh', candidate_activation='tanh', dtype='float32', name=None)

LSTM，即Long-Short Term Memory(长短期记忆)运算。

默认实现方式为diagonal/peephole连接(https://arxiv.org/pdf/1402.1128.pdf)，公式如下：


.. math::
      i_t=\sigma (W_{ix}x_{t}+W_{ih}h_{t-1}+W_{ic}c_{t-1}+b_i)
.. math::
      f_t=\sigma (W_{fx}x_{t}+W_{fh}h_{t-1}+W_{fc}c_{t-1}+b_f)
.. math::
      \widetilde{c_t}=act_g(W_{ct}x_{t}+W_{ch}h_{t-1}+b_{c})
.. math::
      o_t=\sigma (W_{ox}x_{t}+W_{oh}h_{t-1}+W_{oc}c_{t}+b_o)
.. math::
      c_t=f_t\odot c_{t-1}+i_t\odot \widetilde{c_t}
.. math::
      h_t=o_t\odot act_h(c_t)

W 代表了权重矩阵(weight matrix)，例如 :math:`W_{xi}` 是从输入门（input gate）到输入的权重矩阵, :math:`W_{ic}` ，:math:`W_{fc}` ，  :math:`W_{oc}` 是对角权重矩阵(diagonal weight matrix)，用于peephole连接。在此实现方式中，我们使用向量来代表这些对角权重矩阵。

其中：
      - :math:`b` 表示bias向量（ :math:`b_i` 是输入门的bias向量）
      - :math:`σ` 是非线性激励函数（non-linear activations），比如逻辑sigmoid函数
      - :math:`i` ，:math:`f` ，:math:`o` 和 :math:`c` 分别为输入门(input gate)，遗忘门(forget gate)，输出门（output gate）,以及神经元激励向量（cell activation vector）这些向量和神经元输出激励向量（cell output activation vector） :math:`h` 有相同的大小。
      - :math:`⊙` 意为按元素将两向量相乘
      - :math:`act_g` , :math:`act_h` 分别为神经元(cell)输入、输出的激励函数(activation)。常常使用tanh函数。
      - :math:`\widetilde{c_t}` 也被称为候选隐藏状态(candidate hidden state)。可根据当前输入和之前的隐藏状态计算而得

将 ``use_peepholes`` 设为False来禁用 peephole 连接方法。 公式等详细信息请参考 http://www.bioinf.jku.at/publications/older/2604.pdf 。

注意， :math:`W_{xi}x_t, W_{xf}x_t, W_{xc}x_t,W_{xo}x_t` 这些在输入 :math:`x_t` 上的操作不包括在此运算中。用户可以在LSTM operator之前选择使用全连接运算。




参数:
  - **input** (Variable) (LoDTensor) - LodTensor类型，支持variable time length input sequence（时长可变的输入序列）。 该LoDTensor中底层的tensor是一个形为(T X 4D)的矩阵，其中T为此mini-batch上的总共时间步数。D为隐藏层的大小、规模(hidden size)
  - **size** (int) – 4 * 隐藏层大小
  - **h_0** (Variable) – 最初的隐藏状态（hidden state），可选项。默认值为0。它是一个(N x D)张量，其中N是batch大小，D是隐藏层大小。
  - **c_0** (Variable) – 最初的神经元状态（cell state）， 可选项。 默认值0。它是一个(N x D)张量, 其中N是batch大小。h_0和c_0仅可以同时为None，不能只其中一个为None。
  - **param_attr** (ParamAttr|None) – 可学习的隐藏层权重的参数属性。
    注意：
                      - Weights = :math:`\{W_{ch}, W_{ih},  W_{fh},  W_{oh} \}`
                      - 形为(D x 4D), 其中D是hidden size（隐藏层规模）

    如果它被设为None或者 ``ParamAttr`` 属性之一, dynamic_lstm会创建 ``ParamAttr`` 对象作为param_attr。如果没有对param_attr初始化（即构造函数没有被设置）， Xavier会负责初始化参数。默认为None。
  - **bias_attr** (ParamAttr|None) – 可学习的bias权重的属性, 包含两部分，input-hidden bias weights（输入隐藏层的bias权重）和 peephole connections weights（peephole连接权重）。如果 ``use_peepholes`` 值为 ``True`` ， 则意为使用peephole连接的权重。
    另外：
      - use_peepholes = False - Biases = :math:`\{ b_c,b_i,b_f,b_o \}` - 形为(1 x 4D)。
      - use_peepholes = True - Biases = :math:`\{ b_c,b_i,b_f,b_o,W_{ic},W_{fc},W_{oc} \}` - 形为 (1 x 7D)。

    如果它被设为None或 ``ParamAttr`` 的属性之一， ``dynamic_lstm`` 会创建一个 ``ParamAttr`` 对象作为bias_attr。 如果没有对bias_attr初始化（即构造函数没有被设置），bias会被初始化为0。默认值为None。
  - **use_peepholes** (bool) – （默认: True） 是否使用diagonal/peephole连接方式
  - **is_reverse** (bool) – （默认: False） 是否计算反LSTM(reversed LSTM)
  - **gate_activation** (str) – （默认: "sigmoid"）应用于input gate（输入门），forget gate（遗忘门）和 output gate（输出门）的激励函数（activation），默认为sigmoid
  - **cell_activation** (str) – （默认: tanh）用于神经元输出的激励函数(activation), 默认为tanh
  - **candidate_activation** (str) – （默认: tanh）candidate hidden state（候选隐藏状态）的激励函数(activation), 默认为tanh 
  - **dtype** (str) – 即 Data type（数据类型）。 可以选择 [“float32”, “float64”]，默认为“float32”
  - **name** (str|None) – 该层的命名，可选项。如果值为None, 将会自动对该层命名

返回：隐藏状态（hidden state），LSTM的神经元状态。两者都是（T x D）形，且LoD保持与输入一致

返回类型: 元组（tuple）


**代码示例**

..  code-block:: python

  hidden_dim = 512
  forward_proj = fluid.layers.fc(input=input_seq, size=hidden_dim * 4,
                                 bias_attr=False)
  forward, _ = fluid.layers.dynamic_lstm(
      input=forward_proj, size=hidden_dim * 4, use_peepholes=False)













.. _cn_api_fluid_layers_dynamic_lstmp:

dynamic_lstmp
-------------------------------
.. py:function:: paddle.fluid.layers.dynamic_lstmp(input, size, proj_size, param_attr=None, bias_attr=None, use_peepholes=True, is_reverse=False, gate_activation='sigmoid', cell_activation='tanh', candidate_activation='tanh', proj_activation='tanh', dtype='float32', name=None)

动态LSTMP层(Dynamic LSTMP Layer)

LSTMP层(具有循环映射的LSTM)在LSTM层后有一个分离的映射层，从原始隐藏状态映射到较低维的状态，用来减少参数总数，减少LSTM计算复杂度，特别是输出单元相对较大的情况下。 `Long Short-Term Memory Recurrent Neural Network Architectures for Large Scale Acoustic Modeling <https://research.google.com/pubs/archive/43905.pdf>`_

公式如下：

.. math::

        i_t & = \sigma(W_{ix}x_{t} + W_{ir}r_{t-1} + W_{ic}c_{t-1} + b_i)\\
        f_t & = \sigma(W_{fx}x_{t} + W_{fr}r_{t-1} + W_{fc}c_{t-1} + b_f)\\
        \tilde{c_t} & = act_g(W_{cx}x_t + W_{cr}r_{t-1} + b_c)\\
        o_t & = \sigma(W_{ox}x_{t} + W_{or}r_{t-1} + W_{oc}c_t + b_o)\\
        c_t & = f_t \odot c_{t-1} + i_t \odot \tilde{c_t}\\
        h_t & = o_t \odot act_h(c_t)\\
        r_t & = \overline{act_h}(W_{rh}h_t)\\


在以上公式中：
    - :math:`W` : 代表权重矩阵（例如 :math:`W_{xi}` 是输入门道输入的权重矩阵）
    - :math:`W_{ic}` , :math:`W_{fc}` , :math:`W_{oc}`  : peephole connections的对角权重矩阵。在我们的实现中，外面用向量代表这些对角权重矩阵
    - :math:`b` : 代表偏差向量（例如 :math:`b_{i}` 是输入偏差向量）
    - :math:`\delta` : 激活函数，比如逻辑回归函数
    - :math:`i,f,o` 和 :math:`c` :分别代表输入门，遗忘门,输出门和cell激活函数向量，四者的大小和cell输出激活函数向量 :math:`h` 的四者大小相等
    - :math:`h` : 隐藏状态
    - :math:`r` : 隐藏状态的循环映射
    - :math:`\tilde{c_t}` : 候选隐藏状态
    - :math:`\odot` : 向量的元素状态生成
    - :math:`act_g` 和 :math:`act_h` : cell输入和cell输出激活函数，通常使用 :math:`tanh`
    - :math:`\overline{act_h}` : 映射输出的激活函数，通常用 :math:`identity` 或等同的 :math:`act_h` 

将 ``use_peepholes`` 设置为False，断开窥视孔连接（peephole connection）。在此省略公式，详情请参照论文 `LONG SHORT-TERM MEMORY <http://www.bioinf.jku.at/publications/older/2604.pdf>`_ 。

注意输入 :math:`x_{t}` 中的 :math:`W_{xi}x_{t},W_{xf}x_{t},W_{xc}x_{t},W_{xo}x_{t}` 不在此操作符中。用户选择在LSTMP层之前使用全链接层。

参数：
    - **input** (Variable) - dynamic_lstmp层的输入，支持输入序列长度为变量的倍数。该变量的张量为一个矩阵，维度为（T X 4D），T为mini-batch的总时间步长，D是隐藏大小。
    - **size** (int) - 4*隐藏状态大小（hidden size）
    - **proj_size** (int) - 投影输出的大小
    - **param_attr** (ParamAttr|None) -   可学习hidden-hidden权重和投影权重的参数属性。
      说明:
        - Hidden-hidden （隐藏状态到隐藏状态）权重 = :math:`\{ W_{ch},W_{ih},W_{fh},W_{oh} \}`
        - hidden-hidden权重的权重矩阵为（P*4D），P是投影大小，D是隐藏大小。
        - 投影（Projection）权重 = :math:`\{ W_{rh} \}`
        - 投影权重的shape为（D\*P）

      如果设为None或者ParamAttr的一个属性，dynamic_lstm将创建ParamAttr为param_attr。如果param_attr的初始函数未设置，参数则初始化为Xavier。默认:None。
    - **bias_attr** (ParamAttr|None) - 可学习bias权重的bias属性，包含输入隐藏的bias权重和窥视孔连接权重（peephole connection）,前提是use_peepholes设为True。

      说明:
        1.use_peepholes = False
            - Biases = { :math:`b_{c},b_{i},b_{f},b_{o}`}.
            - 维度为（1*4D）

        2.use_peepholes = True
            - Biases = { :math:`b_{c},b_{i},b_{f},b_{o},W_{ic},W_{fc},W_{oc}`}
            - 维度为（1*7D）
        
        如果设置为None或者ParamAttr的一个属性，dynamic_lstm将创建ParamAttr为bias_attr。bias_attr的初始函数未设置，bias则初始化为0.默认：None。
        
    - **use_peepholes** (bool) - 是否开启诊断/窥视孔链接，默认为True。
    - **is_reverse** (bool) - 是否计算反向LSTM，默认为False。
    - **gate_activation** (bool) - 输入门（input gate）、遗忘门（forget gate）和输出门（output gate）的激活函数。Choices = [“sigmoid”，“tanh”，“relu”，“identity”]，默认“sigmoid”。
    - **cell_activation** (str) - cell输出的激活函数。Choices = [“sigmoid”，“tanh”，“relu”，“identity”]，默认“tanh”。
    - **candidate_activation** (str) - 候选隐藏状态（candidate hidden state）的激活状态。Choices = [“sigmoid”，“tanh”，“relu”，“identity”]，默认“tanh”。
    - **proj_activation** (str) - 投影输出的激活函数。Choices = [“sigmoid”，“tanh”，“relu”，“identity”]，默认“tanh”。
    - **dtype** (str) - 数据类型。Choices = [“float32”，“float64”]，默认“float32”。
    - **name** (str|None) - 该层名称（可选）。若设为None，则自动为该层命名。 

返回：含有两个输出变量的元组，隐藏状态（hidden state）的投影和LSTMP的cell状态。投影的shape为（T*P），cell state的shape为（T*D），两者的LoD和输入相同。

返回类型：元组(tuple)

**代码示例**：

.. code-block:: python

    dict_dim, emb_dim = 128, 64
    data = fluid.layers.data(name='sequence', shape=[1],
                         dtype='int32', lod_level=1)
    emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])
    hidden_dim, proj_dim = 512, 256
    fc_out = fluid.layers.fc(input=emb, size=hidden_dim * 4,
                         act=None, bias_attr=None)
    proj_out, _ = fluid.layers.dynamic_lstmp(input=fc_out,
                                         size=hidden_dim * 4,
                                         proj_size=proj_dim,
                                         use_peepholes=False,
                                         is_reverse=True,
                                         cell_activation="tanh",
                                         proj_activation="tanh")











.. _cn_api_fluid_layers_edit_distance:


edit_distance
-------------------------------

.. py:function:: Paddle.fluid.layers.edit_distance(input,label,normalized=True,ignored_tokens=None)

编辑距离运算符

计算一批给定字符串及其参照字符串间的编辑距离。编辑距离也称Levenshtein距离，通过计算从一个字符串变成另一个字符串所需的最少操作步骤来衡量两个字符串的相异度。这里的操作包括插入、删除和替换。

比如给定假设字符串A=“kitten”和参照字符串B=“sitting”，从A变换成B编辑距离为3，至少需要两次替换和一次插入：

“kitten”->“sitten”->“sittn”->“sitting”

输入为LoDTensor,包含假设字符串（带有表示批尺寸的总数）和分离信息（具体为LoD信息）。并且批尺寸大小的参照字符串和输入LoDTensor的顺序保持一致。

输出包含批尺寸大小的结果，代表一对字符串中每个字符串的编辑距离。如果Attr(normalized)为真，编辑距离则处以参照字符串的长度。

参数：
    - **input** (Variable)-假设字符串的索引
    - **label** (Variable)-参照字符串的索引
    - **normalized** (bool,默认为True)-表示是否用参照字符串的长度进行归一化
    - **ignored_tokens** (list<int>,默认为None)-计算编辑距离前需要移除的token
    - **name** (str)-该层名称，可选

返回：[batch_size,1]中序列到序列到编辑距离

返回类型：变量

**代码示例**：

.. code-block:: python

    x = fluid.layers.data(name='x', shape=[1], dtype='float32')
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')
    cost = fluid.layers.edit_distance(input=x,label=y)









.. _cn_api_fluid_layers_elementwise_add:

elementwise_add
-------------------------------

.. py:function:: paddle.fluid.layers.elementwise_add(x, y, axis=-1, act=None, name=None)

逐元素相加算子

等式为：

.. math::
        Out = X + Y

- :math:`X` ：任意维度的张量（Tensor）.
- :math:`Y` ：一个维度必须小于等于X维度的张量（Tensor）。
对于这个运算算子有2种情况：
        1. :math:`Y` 的形状（shape）与 :math:`X` 相同。
        2. :math:`Y` 的形状（shape）是 :math:`X` 的连续子序列。
        
对于情况2:
        1. 用 :math:`Y` 匹配 :math:`X` 的形状（shape），则 ``axis`` 为 :math:`Y` 传到 :math:`X` 上的起始维度索引。
        2. 如果 ``axis`` 为-1（默认值），则 :math:`axis= rank(X)-rank(Y)` 。
        3. 考虑到子序列， :math:`Y` 的大小为1的尾部尺寸将被忽略，例如shape（Y）=（2,1）=>（2）。
        
例如：

..  code-block:: python

        shape(X) = (2, 3, 4, 5), shape(Y) = (,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
        shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
        shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
        shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0

输入 :math:`X` 和 :math:`Y` 可以携带不同的LoD信息。但输出仅与输入 :math:`X` 共享LoD信息。

参数：
        - **x** （Tensor）- 元素op的第一个输入张量（Tensor）。
        - **y** （Tensor）- 元素op的第二个输入张量（Tensor）。
        - **axis** （INT）- （int，默认-1）。将Y传到X上的起始维度索引。
        - **use_mkldnn** （BOOLEAN）- （bool，默认为false）。由 ``MKLDNN`` 使用。
        - **act** （basestring | None）- 激活应用于输出。
        - **name** （basestring | None）- 输出的名称。

返回：        元素运算的输出。









.. _cn_api_fluid_layers_elementwise_div:

elementwise_div
-------------------------------

.. py:function:: paddle.fluid.layers.elementwise_div(x, y, axis=-1, act=None, name=None)

逐元素相除算子

等式是：

.. math::
        Out = X / Y

- :math:`X` ：任何尺寸的张量（Tensor）。
- :math:`Y` ：尺寸必须小于或等于X尺寸的张量（Tensor）。

此运算算子有两种情况：
        1. :math:`Y` 的形状（shape）与 :math:`X` 相同。
        2. :math:`Y` 的形状（shape）是 :math:`X` 的连续子序列。

对于情况2：
        1. 用 :math:`Y` 匹配 :math:`X` 的形状（shape），其中 ``axis`` 将是 :math:`Y` 传到 :math:`X` 上的起始维度索引。
        2. 如果 ``axis`` 为-1（默认值），则 :math:`axis = rank（X）-rank（Y）` 。
        3. 考虑到子序列， :math:`Y` 的大小为1的尾随尺寸将被忽略，例如shape（Y）=（2,1）=>（2）。

例如：

..  code-block:: python

        shape(X) = (2, 3, 4, 5), shape(Y) = (,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
        shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
        shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
        shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0
       
输入 :math:`X` 和 :math:`Y` 可以携带不同的LoD信息。但输出仅与输入 :math:`X` 共享LoD信息。

参数：
        - **x** （Tensor）- 元素op的第一个输入张量（Tensor）。
        - **y** （Tensor）- 元素op的第二个输入张量（Tensor）。
        - **axis** （INT）- （int，默认-1）。将Y传到X上的起始维度索引。
        - **use_mkldnn** （BOOLEAN）- （bool，默认为false）。由MKLDNN使用。
        - **act** （basestring | None）- 激活应用于输出。
        - **name** （basestring | None）- 输出的名称。

返回：        元素运算的输出。
        
        








.. _cn_api_fluid_layers_elementwise_max:

elementwise_max
-------------------------------

.. py:function:: paddle.fluid.layers.elementwise_max(x, y, axis=-1, act=None, name=None)
最大元素算子

等式是：

.. math::
        Out = max(X, Y)
        
- :math:`X` ：任何尺寸的张量（Tensor）。
- :math:`Y` ：尺寸必须小于或等于X尺寸的张量（Tensor）。

此运算算子有两种情况：
        1. :math:`Y` 的形状（shape）与 :math:`X` 相同。
        2. :math:`Y` 的形状（shape）是 :math:`X` 的连续子序列。

对于情况2：
        1. 用 :math:`Y` 匹配 :math:`X` 的形状（shape），其中 ``axis`` 将是 :math:`Y` 传到 :math:`X` 上的起始维度索引。
        2. 如果 ``axis`` 为-1（默认值），则 :math:`axis = rank（X）-rank（Y）` 。
        3. 考虑到子序列， :math:`Y` 的大小为1的尾随尺寸将被忽略，例如shape（Y）=（2,1）=>（2）。

例如：

..  code-block:: python

        shape(X) = (2, 3, 4, 5), shape(Y) = (,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
        shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
        shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
        shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0
        
输入X和Y可以携带不同的LoD信息。但输出仅与输入X共享LoD信息。

参数：
        - **x** （Tensor）- 元素op的第一个输入张量（Tensor）。
        - **y** （Tensor）- 元素op的第二个输入张量（Tensor）。
        - **axis** （INT）- （int，默认-1）。将Y传到X上的起始维度索引。
        - **use_mkldnn** （BOOLEAN）- （bool，默认为false）。由MKLDNN使用。
        - **act** （basestring | None）- 激活应用于输出。
        - **name** （basestring | None）- 输出的名称。

返回：        元素运算的输出。        
        









.. _cn_api_fluid_layers_elementwise_min:

elementwise_min
-------------------------------

.. py:function:: paddle.fluid.layers.elementwise_min(x, y, axis=-1, act=None, name=None)

最小元素算子

等式是：

.. math::
        Out = min(X, Y)
        
- :math:`X` ：任何维数的张量（Tensor）。
- :math:`Y` ：维数必须小于或等于X维数的张量（Tensor）。

此运算算子有两种情况：
        1. :math:`Y` 的形状（shape）与 :math:`X` 相同。
        2. :math:`Y` 的形状（shape）是 :math:`X` 的连续子序列。

对于情况2：
        1. 用 :math:`Y` 匹配 :math:`X` 的形状（shape），其中 ``axis`` 将是 :math:`Y` 传到 :math:`X` 上的起始维度索引。
        2. 如果 ``axis`` 为-1（默认值），则 :math:`axis = rank（X）-rank（Y）` 。
        3. 考虑到子序列， :math:`Y` 的大小为1的尾随尺寸将被忽略，例如shape（Y）=（2,1）=>（2）。

例如：

..  code-block:: python

        shape(X) = (2, 3, 4, 5), shape(Y) = (,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
        shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
        shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
        shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0
        
输入X和Y可以携带不同的LoD信息。但输出仅与输入X共享LoD信息。

参数：
        - **x** （Tensor）- 元素op的第一个输入张量（Tensor）。
        - **y** （Tensor）- 元素op的第二个输入张量（Tensor）。
        - **axis** （INT）- （int，默认-1）。将Y传到X上的起始维度索引。
        - **use_mkldnn** （BOOLEAN）- （bool，默认为false）。由MKLDNN使用。
        - **act** （basestring | None）- 激活应用于输出。
        - **name** （basestring | None）- 输出的名称。

返回：        元素运算的输出。   
 
 








.. _cn_api_fluid_layers_elementwise_mul:

elementwise_mul
-------------------------------

.. py:function:: paddle.fluid.layers.elementwise_mul(x, y, axis=-1, act=None, name=None)

逐元素相乘算子

等式是：

.. math::
        Out = X \odot Y
        
- **X** ：任何尺寸的张量（Tensor）。
- **Y** ：尺寸必须小于或等于X尺寸的张量（Tensor）。

此运算算子有两种情况：
        1. :math:`Y` 的形状（shape）与 :math:`X` 相同。
        2. :math:`Y` 的形状（shape）是 :math:`X` 的连续子序列。

对于情况2：
        1. 用 :math:`Y` 匹配 :math:`X` 的形状（shape），其中 ``axis`` 将是 :math:`Y` 传到 :math:`X` 上的起始维度索引。
        2. 如果 ``axis`` 为-1（默认值），则 :math:`axis = rank（X）-rank（Y）` 。
        3. 考虑到子序列， :math:`Y` 的大小为1的尾随尺寸将被忽略，例如shape（Y）=（2,1）=>（2）。
        
例如：

..  code-block:: python

        shape(X) = (2, 3, 4, 5), shape(Y) = (,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
        shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
        shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
        shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0
        
输入X和Y可以携带不同的LoD信息。但输出仅与输入X共享LoD信息。

参数：
        - **x** - （Tensor），元素op的第一个输入张量（Tensor）。
        - **y** - （Tensor），元素op的第二个输入张量（Tensor）。
        - **axis** （INT）- （int，默认-1）。将Y传到X上的起始维度索引。
        - **use_mkldnn** （BOOLEAN）- （bool，默认为false）。由MKLDNN使用。
        - **act** （basestring | None）- 激活应用于输出。
        - **name** （basestring | None）- 输出的名称。

返回：        元素运算的输出。        
        








.. _cn_api_fluid_layers_elementwise_pow:

elementwise_pow
-------------------------------

.. py:function:: paddle.fluid.layers.elementwise_pow(x, y, axis=-1, act=None, name=None)

逐元素幂运算算子

等式是：

.. math::
        Out = X ^ Y
       
- :math:`X` ：任何维数的张量（Tensor）。
- :math:`Y` ：维数必须小于或等于X维数的张量（Tensor）。

此运算算子有两种情况：
        1. :math:`Y` 的形状（shape）与 :math:`X` 相同。
        2. :math:`Y` 的形状（shape）是 :math:`X` 的连续子序列。

对于情况2：
        1. 用 :math:`Y` 匹配 :math:`X` 的形状（shape），其中 ``axis`` 将是 :math:`Y` 传到 :math:`X` 上的起始维度索引。
        2. 如果 ``axis`` 为-1（默认值），则 :math:`axis = rank（X）-rank（Y）` 。
        3. 考虑到子序列， :math:`Y` 的大小为1的尾随尺寸将被忽略，例如shape（Y）=（2,1）=>（2）。

例如：

..  code-block:: python

        shape(X) = (2, 3, 4, 5), shape(Y) = (,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
        shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
        shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
        shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0
        
输入X和Y可以携带不同的LoD信息。但输出仅与输入X共享LoD信息。

参数：
        - **x** （Tensor）- 元素op的第一个输入张量（Tensor）。
        - **y** （Tensor）- 元素op的第二个输入张量（Tensor）。
        - **axis** （INT）- （int，默认-1）。将Y传到X上的起始维度索引。
        - **use_mkldnn** （BOOLEAN）- （bool，默认为false）。由MKLDNN使用。
        - **act** （basestring | None）- 激活应用于输出。
        - **name** （basestring | None）- 输出的名称。

返回：        元素运算的输出。   
        









.. _cn_api_fluid_layers_elementwise_sub:

elementwise_sub
-------------------------------

.. py:function:: paddle.fluid.layers.elementwise_sub(x, y, axis=-1, act=None, name=None)

逐元素相减算子

等式是：

.. math::
       Out = X - Y
        
- **X** ：任何尺寸的张量（Tensor）。
- **Y** ：尺寸必须小于或等于**X**尺寸的张量（Tensor）。

此运算算子有两种情况：
        1. :math:`Y` 的形状（shape）与 :math:`X` 相同。
        2. :math:`Y` 的形状（shape）是 :math:`X` 的连续子序列。

对于情况2：
        1. 用 :math:`Y` 匹配 :math:`X` 的形状（shape），其中 ``axis`` 将是 :math:`Y` 传到 :math:`X` 上的起始维度索引。
        2. 如果 ``axis`` 为-1（默认值），则 :math:`axis = rank（X）-rank（Y）` 。
        3. 考虑到子序列， :math:`Y` 的大小为1的尾随尺寸将被忽略，例如shape（Y）=（2,1）=>（2）。
        
例如：

..  code-block:: python

        shape(X) = (2, 3, 4, 5), shape(Y) = (,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
        shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
        shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
        shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0
        
输入X和Y可以携带不同的LoD信息。但输出仅与输入X共享LoD信息。

参数：
        - **x** - （Tensor），元素op的第一个输入张量（Tensor）。
        - **y** - （Tensor），元素op的第二个输入张量（Tensor）。
        - **axis** （INT）- （int，默认-1）。将Y传到X上的起始维度索引。
        - **use_mkldnn** （BOOLEAN）- （bool，默认为false）。由MKLDNN使用。
        - **act** （basestring | None）- 激活应用于输出。
        - **name** （basestring | None）- 输出的名称。

返回：        元素运算的输出。
        








.. _cn_api_fluid_layers_elu:

elu
-------------------------------

.. py:function:: paddle.fluid.layers.elu(x, alpha=1.0, name=None)

ELU激活层（ELU Activation Operator）

根据 https://arxiv.org/abs/1511.07289 对输入张量中每个元素应用以下计算。
    
.. math::      
        \\out=max(0,x)+min(0,α∗(ex−1))\\

参数:
    - x(Variable)- ELU operator的输入
    - alpha(FAOAT|1.0)- ELU的alpha值
    - name (str|None) -这个层的名称(可选)。如果设置为None，该层将被自动命名。

返回: ELU操作符的输出

返回类型: 输出(Variable)

**代码示例：**

.. code-block:: python

    x = fluid.layers.data(name="x", shape=[3,10,32,32], dtype="float32")
    y = fluid.layers.elu(x, alpha=0.2)







.. _cn_api_fluid_layers_embedding:

embedding
-------------------------------

.. py:function:: paddle.fluid.layers.embedding(input, size, is_sparse=False, is_distributed=False, padding_idx=None, param_attr=None, dtype='float32')

嵌入层(Embedding Layer)

该层用于查找由输入提供的id在查找表中的嵌入矩阵。查找的结果是input里每个ID对应的嵌入矩阵。
所有的输入变量都作为局部变量传入LayerHelper构造器

参数：
    - **input** (Variable)-包含IDs的张量
    - **size** (tuple|list)-查找表参数的维度。应当有两个参数，一个代表嵌入矩阵字典的大小，一个代表每个嵌入向量的大小。
    - **is_sparse** (bool)-代表是否用稀疏更新的标志
    - **is_distributed** (bool)-是否从远程参数服务端运行查找表
    - **padding_idx** (int|long|None)-如果为 ``None`` ，对查找结果无影响。如果padding_idx不为空，表示一旦查找表中找到input中对应的 ``padding_idz``，则用0填充输出结果。如果 :math:`padding_{i}dx<0` ,在查找表中使用的 ``padding_idx`` 值为 :math:`size[0]+dim` 。
    - **param_attr** (ParamAttr)-该层参数
    - **dtype** (np.dtype|core.VarDesc.VarType|str)-数据类型：float32,float_16,int等。

返回：张量，存储已有输入的嵌入矩阵。

返回类型：变量(Variable)

**代码示例**:

.. code-block:: python

    dict_size = len(dataset.ids)
    data = fluid.layers.data(name='ids', shape=[32, 32], dtype='float32')
    fc = fluid.layers.embedding(input=data, size=[dict_size, 16])









.. _cn_api_fluid_layers_expand:

expand
-------------------------------

.. py:function:: paddle.fluid.layers.expand(x, expand_times, name=None)

expand运算会按给定的次数对输入各维度进行复制（tile）运算。 您应该通过提供属性 ``expand_times`` 来为每个维度设置次数。 X的秩应该在[1,6]中。请注意， ``expand_times`` 的大小必须与X的秩相同。以下是一个用例：

::

        输入(X) 是一个形状为[2, 3, 1]的三维张量（Tensor）:

                [
                   [[1], [2], [3]],
                   [[4], [5], [6]]
                ]

        属性(expand_times):  [1, 2, 2]

        输出(Out) 是一个形状为[2, 6, 2]的三维张量（Tensor）:

                [
                    [[1, 1], [2, 2], [3, 3], [1, 1], [2, 2], [3, 3]],
                    [[4, 4], [5, 5], [6, 6], [4, 4], [5, 5], [6, 6]]
                ]
 
参数:
        - **x** (Variable)- 一个秩在[1, 6]范围中的张量（Tensor）.
        - **expand_times** (list|tuple) - 每一个维度要扩展的次数.
        
返回：     expand变量是LoDTensor。expand运算后，输出（Out）的每个维度的大小等于输入（X）的相应维度的大小乘以 ``expand_times`` 给出的相应值。

返回类型：   变量（Variable）

**代码示例**

..  code-block:: python

        x = fluid.layers.data(name='x', shape=[10], dtype='float32')
        out = fluid.layers.expand(x=x, expand_times=[1, 2, 2])
               
               








.. _cn_api_fluid_layers_fc:

fc
-------------------------------

.. py:function::  paddle.fluid.layers.fc(input, size, num_flatten_dims=1, param_attr=None, bias_attr=None, act=None, is_test=False, name=None)


**全连接层**

该函数在神经网络中建立一个全连接层。 它可以同时将多个tensor作为自己的输入，并为每个输入的tensor创立一个变量，称为“权”（weights），等价于一个从每个输入单元到每个输出单元的全连接权矩阵。FC层用每个tensor和它对应的权相乘得到输出tensor。如果有多个输入tensor，那么多个乘法运算将会加在一起得出最终结果。如果 ``bias_attr`` 非空，则会新创建一个偏向变量（bias variable），并把它加入到输出结果的运算中。最后，如果 ``act`` 非空，它也会加入最终输出的计算中。

这个过程可以通过如下公式表现：

.. math::

        \\Out=Act(\sum^{N-1}_{i=0}X_iW_i+b) \\


上述等式中：
  - :math:`N` ：输入tensor的数目
  - :math:`X_i` : 输入的tensor
  - :math:`W` ：该层创立的权
  - :math:`b` ：该层创立的bias参数
  - :math:`Act` : activation function(激励函数)
  - :math:`Out` : 输出tensor


参数:
  - **input** (Variable|list of Variable) – 该层的输入tensor(s)（张量），其维度至少是2
  - **size** (int) – 该层输出单元的数目
  - **num_flatten_dims** (int, default 1) – fc层可以接受一个维度大于2的tensor。此时， 它首先会被扁平化(flattened)为一个二维矩阵。 参数``num_flatten_dims`` 决定了输入tensor的flattened方式: 前 ``num_flatten_dims`` (包含边界，从1开始数) 个维度会被扁平化为最终矩阵的第一维 (维度即为矩阵的高), 剩下的 rank(X) - num_flatten_dims 维被扁平化为最终矩阵的第二维 (即矩阵的宽)。 例如， 假设X是一个五维tensor，其形可描述为(2, 3, 4, 5, 6), 且num_flatten_dims = 3。那么扁平化的矩阵形状将会如此： (2 x 3 x 4, 5 x 6) = (24, 30)
  - **param_attr** (ParamAttr|list of ParamAttr, default None) – 该层可学习的参数/权的参数属性
  - **bias_attr** (ParamAttr|list of ParamAttr, default None) – 该层bias变量的参数属性。如果值为False， 则bias变量不参与输出单元运算。 如果值为None，bias变量被初始化为0。默认为 None。
  - **act** (str, default None) – 应用于输出的Activation（激励函数）
  - **is_test** (bool) – 表明当前执行是否处于测试阶段的标志
  - **name** (str, default None) – 该层的命名


返回：转换结果

返回类型: Variable

弹出异常：``ValueError`` - 如果输入tensor的维度小于2

**代码示例**

..  code-block:: python

        data = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
        fc = fluid.layers.fc(input=data, size=1000, act="tanh")














.. _cn_api_fluid_layers_flatten:

flatten
-------------------------------

将输入张量压扁成二维矩阵

例如：

.. code-block:: text

    Case 1:
      
      给定
        X.shape = (3, 100, 100, 4)
      且
        axis = 2
      得到:
        Out.shape = (3 * 100, 4 * 100)
    
    Case 2:
      
      给定
        X.shape = (3, 100, 100, 4)
      且
        axis = 0
      得到:
        Out.shape = (1, 3 * 100 * 100 * 4)

参数：
	- **x** (Variable): 一个秩>=axis 的张量
	- **axis** (int)：指出应该将哪些输入维度(唯一的) flatten 到输出的外部维度。axis的值必须在[0,R]范围内，其中R是输入张量的秩。当 axis= 0 时，输出张量的形状为(1，(d_0 X d_1…)，其中输入张量的形状为(d_0, d_1，…)d_n)。
	- **name** (str|None)：此层的名称(可选)。如果没有设置，层将自动命名。

返回：Variable: 一个二维张量，它包含输入张量的内容，输入维数将沿给定轴flatten到输出的外部维度，剩余的输入维数flatten到输出的内部维数。

抛出异常：
	- ValueError: 如果 x 不是一个变量
	- ValueError: 如果axis的范围不在 [0, rank(x)]

**代码示例**

.. code-block:: python

    x = fluid.layers.data(name="x", shape=[4, 4, 3], dtype="float32")
    out = fluid.layers.flatten(x=x, axis=2)










.. _cn_api_fluid_layers_gather:

gather
-------------------------------

.. py:function:: paddle.fluid.layers.gather(input, index)

收集层（gather layer）

根据索引index获取X的最外层维度的条目，并将它们串连在一起。

.. math::
                        Out=X[Index]

::

        X = [[1, 2],
             [3, 4],
             [5, 6]]

        Index = [1, 2]

        Then:

        Out = [[3, 4],
               [5, 6]]


参数:
        - **input** (Variable) - input 的rank >= 1。
        - **index** (Variable) - index的rank = 1。
    
返回：	output (Variable)

**代码示例**

..  code-block:: python
        
	output = fluid.layers.gather(x, index)









.. _cn_api_fluid_layers_gaussian_random:

gaussian_random
-------------------------------

.. py:function:: paddle.fluid.layers.gaussian_random(shape, mean=0.0, std=1.0, seed=0, dtype='float32')

gaussian_random算子。

用于使用高斯随机生成器初始化张量（Tensor）。

参数：
        - **shape** （tuple | list）- （vector <int>）随机张量的维数
        - **mean** （Float）- （默认值0.0）随机张量的均值
        - **std** （Float）- （默认值为1.0）随机张量的std
        - **seed** （Int）- （默认值为 0）生成器随机生成种子。0表示使用系统范围的种子。注意如果seed不为0，则此运算符每次将始终生成相同的随机数
        - **dtype** （np.dtype | core.VarDesc.VarType | str）- 输出的数据类型。

返回：        输出高斯随机运算矩阵

返回类型：        输出（Variable）

       








.. _cn_api_fluid_layers_gaussian_random_batch_size_like:

gaussian_random_batch_size_like
-------------------------------

.. py:function:: paddle.fluid.layers.gaussian_random_batch_size_like(input, shape, input_dim_idx=0, output_dim_idx=0, mean=0.0, std=1.0, seed=0, dtype='float32')

用于使用高斯随机发生器初始化张量。分布的defalut均值为0.并且分布的defalut标准差（std）为1.用户可以通过输入参数设置mean和std。

参数：
        - **input** （Variable）- 其input_dim_idx'th维度指定batch_size的张量（Tensor）。
        - **shape** （元组|列表）- 输出的形状。
        - **input_dim_idx** （Int）- 默认值0.输入批量大小维度的索引。
        - **output_dim_idx** （Int）- 默认值0.输出批量大小维度的索引。
        - **mean** （Float）- （默认值0.0）高斯分布的平均值（或中心值）。
        - **std** （Float）- （默认值 1.0）高斯分布的标准差（std或spread）。
        - **seed** （Int）- （默认为0）用于随机数引擎的随机种子。0表示使用系统生成的种子。请注意，如果seed不为0，则此运算符将始终每次生成相同的随机数。
        - **dtype** （np.dtype | core.VarDesc.VarType | str）- 输出数据的类型为float32，float_16，int等。

返回：        指定形状的张量将使用指定值填充。

返回类型：        输出（Variable）。










.. _cn_api_fluid_layers_get_tensor_from_selected_rows:

get_tensor_from_selected_rows
-------------------------------

.. py:function::  paddle.fluid.layers.get_tensor_from_selected_rows(x, name=None)

:code:`Get Tensor From Selected Rows` 用于从选中行（Selected Rows）中获取张量

参数：
	- **x** (Variable) - 输入，类型是SelectedRows
	- **name** (basestring|None) - 输出的名称

返回： 输出类型为LoDTensor

返回类型： out(Variable)











.. _cn_api_fluid_layers_grid_sampler:

grid_sampler
-------------------------------

.. py:function::  paddle.fluid.layers.grid_sampler(x, grid, name=None)

该操作使用基于flow field网格的双线性插值对输入X进行采样，通常由affine_grid生成。

形状为(N、H、W、2)的网格是由两个形状均为(N、H、W)的坐标(grid_x grid_y)连接而成的。

其中，grid_x是输入数据x的第四个维度(宽度维度)的索引，grid_y是第三维度(高维)的索引，最终得到4个最接近的角点的双线性插值值。

step 1：

	得到(x, y)网格坐标，缩放到[0,h -1/W-1]

	grid_x = 0.5 * (grid[:, :, :, 0] + 1) * (W - 1) grid_y = 0.5 * (grid[:, :, :, 1] + 1) * (H - 1)

step 2：

	在每个[H, W]区域用网格(X, y)作为输入数据X的索引，并将双线性插值点值由4个最近的点表示。

.. code-block:: text

      wn ------- y_n ------- en
      |           |           |
      |          d_n          |
      |           |           |
     x_w --d_w-- grid--d_e-- x_e
      |           |           |
      |          d_s          |
      |           |           |
      ws ------- y_s ------- wn
    
    x_w = floor(x)              // west side x coord
    x_e = x_w + 1               // east side x coord
    y_n = floor(y)              // north side y coord
    y_s = y_s + 1               // south side y coord
    d_w = grid_x - x_w          // distance to west side
    d_e = x_e - grid_x          // distance to east side
    d_n = grid_y - y_n          // distance to north side
    d_s = y_s - grid_y          // distance to south side
    wn = X[:, :, y_n, x_w]      // north-west point value
    en = X[:, :, y_n, x_e]      // north-east point value
    ws = X[:, :, y_s, x_w]      // south-east point value
    es = X[:, :, y_s, x_w]      // north-east point value
    

    output = wn * d_e * d_s + en * d_w * d_s
           + ws * d_e * d_n + es * d_w * d_n

参数：
	- **x** (Variable): 输入数据，形状为[N, C, H, W]
	- **grid** (Variable): 输入网格张量，形状为[N, H, W, 2]
	- **name** (str, default None): 该层的名称

返回： **out** (Variable): 输入X基于输入网格的bilnear插值计算结果，形状为[N, C, H, W]

**代码示例：**

.. code-block:: python

    x = fluid.layers.data(name='x', shape=[3, 10, 32, 32], dtype='float32')
    theta = fluid.layers.data(name='theta', shape=[3, 2, 3], dtype='float32')
    grid = fluid.layers.affine_grid(input=theta, size=[3, 10, 32, 32]})
    out = fluid.layers.grid_sampler(x=x, grid=grid)










.. _cn_api_fluid_layers_group_norm:

group_norm
-------------------------------

.. py:function::  paddle.fluid.layers.group_norm(input, groups, epsilon=1e-05, param_attr=None, bias_attr=None, act=None, data_layout='NCHW', name=None)

参考论文： `Group Normalization <https://arxiv.org/abs/1803.08494>`_

参数：
	- **input** (Variable)：输入张量变量
	- **groups** (int)：从 channel 中分离出来的 group 的数目
	- **epsilon** (float)：为防止方差除零，增加一个很小的值
	- **param_attr** (ParamAttr|None)：可学习标度的参数属性 :math:`g`,如果设置为False，则不会向输出单元添加标度。如果设置为0，偏差初始化为1。默认值:None
	- **bias_attr** (ParamAttr|None)：可学习偏置的参数属性 :math:`b ` , 如果设置为False，则不会向输出单元添加偏置量。如果设置为零，偏置初始化为零。默认值:None。
	- **act** (str):将激活应用于输出的 group normalizaiton
	- **data_layout** (string|NCHW): 只支持NCHW。
	- **name** (str):这一层的名称（可选）

返回： Variable: 一个张量变量，它是对输入进行 group normalization 后的结果。

**代码示例：**

.. code-block:: python

    data = fluid.layers.data(name='data', shape=[8, 32, 32],
                             dtype='float32')
    x = fluid.layers.group_norm(input=data, groups=4)










.. _cn_api_fluid_layers_gru_unit:

gru_unit
-------------------------------

.. py:function::  paddle.fluid.layers.gru_unit(input, hidden, size, param_attr=None, bias_attr=None, activation='tanh', gate_activation='sigmoid')

GRU单元层。GRU执行步骤基于如下等式：

.. math::
    u_t=actGate(xu_t+W_{u}h_{t-1}+b_u)
.. math::
    r_t=actGate(xr_t+W_{r}h_{t-1}+b_r)
.. math::
    m_t=actNode(xm_t+W_{c}dot(r_t,h_{t-1})+b_m)
.. math::
    h_t=dot((1-u_t),m_t)+dot(u_t,h_{t-1})
    
GRU单元的输入包括 :math:`z_t` ， :math:`h_{t-1}` 。在上述等式中， :math:`z_t` 会被分割成三部分： :math:`xu_t` 、 :math:`xr_t` 和 :math:`xm_t`  。
这意味着要为一批输入实现一个全GRU层，我们需要采用一个全连接层，才能得到 :math:`z_t=W_{fc}x_t` 。
:math:`u_t` 和 :math:`r_t` 分别代表了GRU神经元的update gates（更新门）和reset gates(重置门)。
和LSTM不同，GRU少了一个门（它没有LSTM的forget gate）。但是它有一个叫做中间候选隐藏状态（intermediate candidate hidden output）的输出，
记为 :math:`m_t` 。 该层有三个输出： :math:`h_t, dot(r_t,h_{t-1})` 以及 :math:`u_t，r_t，m_t` 的连结(concatenation)。
 
 


参数:
  - **input** (Variable) – 经FC层变换后的当前步骤的输入值
  - **hidden** (Variable) –  从上一步而来的gru unit 隐藏状态值(hidden value)
  - **size** (integer) – 输入数据的维度
  - **param_attr** (ParamAttr|None) – 可学习的隐藏层权重矩阵的参数属性。
    注意：
      - 该权重矩阵形为 :math:`(T×3D)` ， :math:`D` 是隐藏状态的规模（hidden size）
      - 该权重矩阵的所有元素由两部分组成， 一是update gate和reset gate的权重，形为 :math:`(D×2D)` ；二是候选隐藏状态（candidate hidden state）的权重矩阵，形为 :math:`(D×D)`
    如果该函数参数值为None或者 ``ParamAttr`` 类中的属性之一，gru_unit则会创建一个 ``ParamAttr`` 类的对象作为 param_attr。如果param_attr没有被初始化，那么会由Xavier来初始化它。默认值为None
  - **bias_attr** (ParamAttr|bool|None) - GRU的bias变量的参数属性。形为 :math:`(1x3D)` 的bias连结（concatenate）在update gates（更新门），reset gates(重置门)以及candidate calculations（候选隐藏状态计算）中的bias。如果值为False，那么上述三者将没有bias参与运算。若值为None或者 ``ParamAttr`` 类中的属性之一，gru_unit则会创建一个 ``ParamAttr`` 类的对象作为 bias_attr。如果bias_attr没有被初始化，那它会被默认初始化为0。默认值为None。
  - **activation** (string) –  神经元 “actNode” 的激励函数（activation）类型。默认类型为‘tanh’
  - **gate_activation** (string) – 门 “actGate” 的激励函数（activation）类型。 默认类型为 ‘sigmoid’
  

返回：	 hidden value（隐藏状态的值），reset-hidden value(重置隐藏状态值)，gate values(门值)

返回类型:	 元组（tuple）


**代码示例**

..  code-block:: python

    # 假设我们现在有x_t_data和size=10的之前的隐藏层
    x_t = fluid.layers.fc(input=x_t_data, size=30)
    hidden_val, r_h_val, gate_val = fluid.layers.gru_unit(input=x_t,
                                          hidden = prev_hidden)














.. _cn_api_fluid_layers_hard_sigmoid:

hard_sigmoid
-------------------------------

.. py:function:: paddle.fluid.layers.hard_sigmoid(x, slope=0.2, offset=0.5, name=None)

HardSigmoid激活算子。

sigmoid的分段线性逼近(https://arxiv.org/abs/1603.00391)，比sigmoid快得多。

.. math::   

      \\out=\max(0,\min(1,slope∗x+shift))\\
 
斜率是正数。偏移量可正可负的。斜率和位移的默认值是根据上面的参考设置的。建议使用默认值。

参数：
    - **x** (Variable) - HardSigmoid operator的输入
    - **slope** (FLOAT|0.2) -斜率
    - **offset** (FLOAT|0.5)  - 偏移量
    - **name** (str|None) - 这个层的名称(可选)。如果设置为None，该层将被自动命名。


**代码示例：**

.. code-block:: python

    x = fluid.layers.data(name="x", shape=[3,10,32,32], dtype="float32")
    y = fluid.layers.hard_sigmoid(x, slope=0.3, offset=0.8)






.. _cn_api_fluid_layers_hash:

hash
-------------------------------

.. py:function::  paddle.fluid.layers.hash(input, hash_size, num_hash=1, name=None)

将输入 hash 到一个整数，该数的值小于给定的 hash size

我们使用的哈希算法是 xxHash - `Extremely fast hash algorithm <https://github.com/Cyan4973/xxHash/tree/v0.6.5>`_

提供一简单的例子：

.. code-block:: text

	给出：

		# shape [2, 2]
		input.data = [
		    [[1], [2]],
		    [[3], [4]],
		]

		input.lod = [[0, 2]]

		hash_size = 10000

		num_hash = 4

	然后:

		哈希操作将这个二维input的所有数字作为哈希算法每次的输入。

		每个输入都将被哈希4次，最终得到一个长度为4的数组。

		数组中的每个值的范围从0到9999。



  	# shape [2, 4]
  	output.data = [
  	    [[9662], [9217], [1129], [8487]],
  	    [[8310], [1327], [1654], [4567]],
  	]

  	output.lod = [[0, 2]]

参数：
	- **input** (Variable) - 输入变量是一个 one-hot 词。输入变量的维数必须是2。
	- **hash_size** (int) - 哈希算法的空间大小。输出值将保持在 :math:`[0, hash\_size - 1]` 范围内。
	- **num_hash** (int) - 哈希次数，默认为1。
	- **name** (str, default None) - 该层的名称

返回：哈希的结果变量，是一个lodtensor。

返回类型： Variable

**代码示例：**

.. code-block:: python

    word_dict = paddle.dataset.imdb.word_dict()
    x = fluid.layers.data(shape[1], dtype='int32', lod_level=1)
    out = fluid.layers.hash(input=x, num_hash=4, hash_size=1000)









.. _cn_api_fluid_layers_hsigmoid:

hsigmoid
-------------------------------

.. py:function:: paddle.fluid.layers.hsigmoid(input, label, num_functiones, param_attr=None, bias_attr=None, name=None, path_table=None, path_code=None, is_custom=False, is_sparse=False)

层次sigmod（ hierarchical sigmoid ）加速语言模型的训练过程。这个operator将类别组织成一个完全二叉树，也可以使用 ``is_custom`` 参数来传入自定义的树结构来实现层次化。 

树中每个叶节点表示一个类(一个单词)，每个内部节点进行一个二分类。对于每个单词，都有一个从根到它的叶子节点的唯一路径，hsigmoid计算路径上每个内部节点的损失（cost），并将它们相加得到总损失（cost）。

hsigmoid可以把时间复杂度 :math:`O(N)` 优化到 :math:`O(logN)` ,其中 :math:`N` 表示单词字典的大小。

使用默认树结构，请参考 `Hierarchical Probabilistic Neural Network Language Model <http://www.iro.umontreal.ca/~lisa/pointeurs/hierarchical-nnlm-aistats05.pdf>`_ 。

若要使用自定义树结构，请设置 ``is_custom`` 值为True。但在此之前，请完成以下几步部署：

1.使用自定义词典来建立二叉树，每个叶结点都应该是词典中的单词

2.建立一个dict类型数据结构，来存储 **单词id -> 该单词叶结点至根节点路径** 的映射，称之为路径表 ``path_table`` 参数

3.建立一个dict类型数据结构，来存储 **单词id -> 该单词叶结点至根节点路径的编码(code)** 的映射。 编码code是指每次二分类的标签，1为真，0为假

4.现在我们的每个单词都已经有自己的路径和路径编码，当对于同一批输入进行操作时，你可以同时传入一批路径和路径编码进行运算。

参数:
    - **input** (Variable) - 输入张量，shape为 ``[N×D]`` ,其中 ``N`` 是minibatch的大小，D是特征大小。
    - **label** (Variable) - 训练数据的标签。该tensor的shape为 ``[N×1]``   
    - **num_classes** (int) - 类别的数量不能少于2。若使用默认树结构，该参数必须用户设置。当 ``is_custom=False`` 时，该项绝不能为None。反之，如果 ``is_custom=True`` ，它取值应为非叶节点的个数，来指明二分类实用的类别数目。
    - **param_attr** (ParamAttr|None) - 可学习参数/ hsigmoid权重的参数属性。如果将其设置为ParamAttr的一个属性或None，则将ParamAttr设置为param_attr。如果没有设置param_attr的初始化器，那么使用用Xavier初始化。默认值:没None。
    - **bias_attr** (ParamAttr|bool|None) - hsigmoid偏置的参数属性。如果设置为False，则不会向输出添加偏置。如果将其设置ParamAttr的一个属性或None，则将ParamAttr设置为bias_attr。如果没有设置bias_attr的初始化器，偏置将初始化为零。默认值:None。
    - **name** (str|None) - 该layer的名称(可选)。如果设置为None，该层将被自动命名。默认值:None。
    - **path_table** (Variable|None) – 存储每一批样本从词到根节点的路径。路径应为从叶至根方向。 ``path_table`` 和 ``path_code`` 应具有相同的形, 对于每个样本 i ，path_table[i]为一个类似np.array的结构，该数组内的每个元素都是其双亲结点权重矩阵的索引
    - **path_code** (Variable|None) – 存储每批样本的路径编码，仍然是按从叶至根方向。各样本路径编码批都由其各祖先结点的路径编码组成
    - **is_custom** (bool|False) – 使用用户自定义二叉树取代默认二叉树结构，如果该项为真， 请务必设置 ``path_table`` , ``path_code`` , ``num_classes`` , 否则就需要设置 num_classes 
    - **is_sparse** (bool|False) – 使用稀疏更新方式，而非密集更新。如果为真， W的梯度和输入梯度将会变得稀疏

返回:  (LoDTensor) 层次sigmod（ hierarchical sigmoid） 。shape[N, 1]
    
返回类型:  Out


**代码示例**

..  code-block:: python
        
	x = fluid.layers.data(name='x', shape=[2], dtype='float32')
    	y = fluid.layers.data(name='y', shape=[1], dtype='int64')
    	out = fluid.layers.hsigmoid(input=x, label=y, num_classes=6)










.. _cn_api_fluid_layers_im2sequence:

im2sequence
-------------------------------

.. py:function:: paddle.fluid.layers.im2sequence(input, filter_size=1, stride=1, padding=0, input_image_size=None, out_stride=1, name=None)

从输入张量中提取图像张量，与im2col相似，shape={input.batch_size * output_height * output_width, filter_size_H * filter_size_W * input.通道}。这个op使用filter / kernel扫描图像并将这些图像转换成序列。一个图片展开后的timestep的个数为output_height * output_width，其中output_height和output_width由下式计算:


.. math:: 
                        output\_size=1+\frac{(2∗padding+img\_size−block\_size+stride-1}{stride}

每个timestep的维度为 :math:`block\_y * block\_x * input.channels` 。

参数:
	- **input** （Variable）- 输入张量，格式为[N, C, H, W]
	- **filter_size** (int|tuple|None) - 滤波器大小。如果filter_size是一个tuple，它必须包含两个整数(filter_size_H, filter_size_W)。否则，过滤器将是一个方阵。
    	- **stride** (int|tuple) - 步长大小。如果stride是一个元组，它必须包含两个整数(stride_H、stride_W)。否则，stride_H = stride_W = stride。默认:stride = 1。
    	- **padding** (int|tuple) - 填充大小。如果padding是一个元组，它可以包含两个整数(padding_H, padding_W)，这意味着padding_up = padding_down = padding_H和padding_left = padding_right = padding_W。或者它可以使用(padding_up, padding_left, padding_down, padding_right)来指示四个方向的填充。否则，标量填充意味着padding_up = padding_down = padding_left = padding_right = padding Default: padding = 0。
    	- **input_image_size** (Variable) - 输入包含图像的实际大小。它的维度为[batchsize，2]。该参数可有可无，是用于batch推理。
    	- **out_stride** (int|tuple) - 通过CNN缩放图像。它可有可无，只有当input_image_size不为空时才有效。如果out_stride是tuple，它必须包含(out_stride_H, out_stride_W)，否则，out_stride_H = out_stride_W = out_stride。
    	- **name** (int) - 该layer的名称，可以忽略。

返回：	LoDTensor shaoe为{batch_size * output_height * output_width, filter_size_H * filter_size_W * input.channels}。如果将输出看作一个矩阵，这个矩阵的每一行都是一个序列的step。

返回类型:	output

::

	Given:

    x = [[[[ 6.  2.  1.]
    	[ 8.  3.  5.]
    	[ 0.  2.  6.]]

        [[ 2.  4.  4.]
         [ 6.  3.  0.]
         [ 6.  4.  7.]]]

       [[[ 6.  7.  1.]
         [ 5.  7.  9.]
         [ 2.  4.  8.]]

        [[ 1.  2.  1.]
         [ 1.  3.  5.]
         [ 9.  0.  8.]]]]

    x.dims = {2, 2, 3, 3}

    And:

    filter = [2, 2]
    stride = [1, 1]
    padding = [0, 0]

    Then:

    output.data = [[ 6.  2.  8.  3.  2.  4.  6.  3.]
                   [ 2.  1.  3.  5.  4.  4.  3.  0.]
                   [ 8.  3.  0.  2.  6.  3.  6.  4.]
                   [ 3.  5.  2.  6.  3.  0.  4.  7.]
                   [ 6.  7.  5.  7.  1.  2.  1.  3.]
                   [ 7.  1.  7.  9.  2.  1.  3.  5.]
                   [ 5.  7.  2.  4.  1.  3.  9.  0.]
                   [ 7.  9.  4.  8.  3.  5.  0.  8.]]

    output.dims = {8, 8}

    output.lod = [[4, 4]]


**代码示例**

..  code-block:: python
  
    output = fluid.layers.im2sequence(
    input=layer, stride=[1, 1], filter_size=[2, 2])










.. _cn_api_fluid_layers_image_resize:

image_resize
-------------------------------

.. py:function:: paddle.fluid.layers.image_resize(input, out_shape=None, scale=None, name=None, resample='BILINEAR')

调整一批图片的大小
    
输入张量的shape为(num_batch, channels, in_h, in_w)，并且调整大小只适用于最后两个维度(高度和宽度)。
    
支持重新取样方法: 

    BILINEAR：双线性插值
    NEAREST：最近邻插值

参数:
    - **input** (Variable) - 图片调整层的输入张量，这是一个shape=4的张量(num_batch, channels, in_h, in_w)
    - **out_shape** (list|tuple|Variable|None) - 图片调整层的输出，shape为(out_h, out_w)。默认值:None
    - **scale** (float|None)-输入的高度或宽度的乘数因子 。 out_shape和scale至少要设置一个。out_shape的优先级高于scale。默认值:None
    - **name** (str|None) - 该层的名称(可选)。如果设置为None，该层将被自动命名
    - **resample** (str) - 重采样方法。目前只支持“双线性”。默认值:双线性插值
    - **actual_shape** (Variable) - 可选输入，用于动态指定输出形状。如果指定actual_shape，图像将根据给定的形状调整大小，而不是根据指定形状的 :code:`out_shape` 和 :code:`scale` 进行调整。也就是说， :code:`actual_shape` 具有最高的优先级。如果希望动态指定输出形状，建议使用 :code:`actual_shape` 而不是 :code:`out_shape` 。在使用actual_shape指定输出形状时，还需要设置out_shape和scale之一，否则在图形构建阶段会出现错误。默认值:None


返回： 4维tensor，shape为 (num_batches, channls, out_h, out_w).

返回类型:	变量（variable）

抛出异常：
	- :code:`TypeError` - out_shape应该是一个列表、元组或变量。
	- :code:`TypeError` - actual_shape应该是变量或None
	- :code:`ValueError` - image_resize的"resample"只能是"BILINEAR"或"NEAREST"。
	- :code:`ValueError` - out_shape 和 scale 不能为 None
	- :code:`ValueError` - out_shape 的长度必须为 2

**代码示例**

..  code-block:: python
        
	out = fluid.layers.image_resize(input, out_shape=[12, 12], resample="NEAREST") 
  











.. _cn_api_fluid_layers_image_resize_short:

image_resize_short
-------------------------------

.. py:function:: paddle.fluid.layers.image_resize_short(input, out_short_len, resample='BILINEAR')

调整一批图片的大小。输入图像的短边将被调整为给定的out_short_len 。输入图像的长边按比例调整大小，最终图像的长宽比保持不变。

参数:
        - **input** (Variable) -  图像调整图层的输入张量，这是一个4维的形状张量(num_batch, channels, in_h, in_w)。
        - **out_short_len** (int) -  输出图像的短边长度。
        - **resample** (str) - resample方法，默认为双线性插值。
    
返回：	4维张量，shape为(num_batch, channls, out_h, out_w)

返回类型:	变量（variable）











.. _cn_api_fluid_layers_l2_normalize:

l2_normalize
-------------------------------

.. py:function:: paddle.fluid.layers.l2_normalize(x,axis,epsilon=1e-12,name=None)

L2正则（L2 normalize Layer）

该层用欧几里得距离之和对维轴的x归一化。对于1-D张量（系数矩阵的维度固定为0），该层计算公式如下：

.. math::

    y=\frac{x}{\sqrt{\sum x^{2}+epsion}}

对于x多维的情况，该函数分别对维度轴上的每个1-D切片单独归一化

参数：
    - **x** (Variable|list)- l2正则层（l2_normalize layer）的输入
    - **axis** (int)-运用归一化的轴。如果轴小于0，归一化的维是rank(X)+axis。-1是最后维
    - **epsilon** (float)-epsilon用于避免分母为0，默认值为1e-10
    - **name** (str|None)-该层名称（可选）。如果设为空，则自动为该层命名
    
    返回：输出张量，同x的维度一致
    
    返回类型：变量
    
**代码示例**：

.. code-block:: python

    data = fluid.layers.data(name="data",
                         shape=(3, 17, 13),
                         dtype="float32")
    normed = fluid.layers.l2_normalize(x=data, axis=1)









.. _cn_api_fluid_layers_label_smooth:

label_smooth
-------------------------------

.. py:function:: paddle.fluid.layers.label_smooth(label, prior_dist=None, epsilon=0.1, dtype='float32', name=None)

标签平滑是一种对分类器层进行正则化的机制，称为标签平滑正则化(LSR)。


由于直接优化正确标签的对数似然可能会导致过拟合，降低模型的适应能力，因此提出了标签平滑的方法来降低模型置信度。
标签平滑使用标签 :math:`y` 自身和一些固定模式随机分布变量 :math:`\mu` 。对 :math:`k` 标签，我们有：

.. math::
            \tilde{y_k} = (1 - \epsilon) * y_k + \epsilon * \mu_k,

其中 :math:`1-\epsilon` 和 :math:`\epsilon` 分别是权重， :math:`\tilde{y_k}` 是平滑后的标签。 通常μ 使用均匀分布


查看更多关于标签平滑的细节 https://arxiv.org/abs/1512.00567

参数：
  - **label** （Variable） - 包含标签数据的输入变量。 标签数据应使用 one-hot 表示。
  - **prior_dist** （Variable） - 用于平滑标签的先验分布。 如果未提供，则使用均匀分布。 prior_dist的shape应为 :math:`(1，class\_num)` 。
  - **epsilon** （float） - 用于混合原始真实分布和固定分布的权重。
  - **dtype** （np.dtype | core.VarDesc.VarType | str） - 数据类型：float32，float_64，int等。
  - **name** （str | None） - 此层的名称（可选）。 如果设置为None，则将自动命名图层。

返回：张量变量, 包含平滑后的标签

返回类型: Variable

**代码示例**

..  code-block:: python

    label = layers.data(name="label", shape=[1], dtype="float32")
    one_hot_label = layers.one_hot(input=label, depth=10)
    smooth_label = layers.label_smooth(
    label=one_hot_label, epsilon=0.1, dtype="float32")









.. _cn_api_fluid_layers_layer_norm:

layer_norm
-------------------------------

.. py:function:: paddle.fluid.layers.layer_norm(input, scale=True, shift=True, begin_norm_axis=1, epsilon=1e-05, param_attr=None, bias_attr=None, act=None, name=None)

假设特征向量存在于维度 ``begin_norm_axis ... rank (input）`` 上，计算大小为 ``H`` 的特征向量a在该维度上的矩统计量，然后使用相应的统计量对每个特征向量进行归一化。 之后，如果设置了 ``scale`` 和 ``shift`` ，则在标准化的张量上应用可学习的增益和偏差以进行缩放和移位。

请参考 `Layer Normalization <https://arxiv.org/pdf/1607.06450v1.pdf>`_ 
            
公式如下

.. math::
            \\\mu=\frac{1}{H}\sum_{i=1}^{H}a_i\\
.. math::
            \\\sigma=\sqrt{\frac{1}{H}\sum_i^H{(a_i-\mu)^2}}\\
.. math::
             \\h=f(\frac{g}{\sigma}(a-\mu) + b)\\
             
- :math:`\alpha` : 该层神经元输入总和的向量表示
- :math:`H` : 层中隐藏的神经元个数
- :math:`g` : 可训练的缩放因子参数
- :math:`b` : 可训练的bias参数


参数:
  - **input** （Variable） - 输入张量变量。
  - **scale** （bool） - 是否在归一化后学习自适应增益g。默认为True。
  - **shift** （bool） - 是否在归一化后学习自适应偏差b。默认为True。
  - **begin_norm_axis** （int） - ``begin_norm_axis`` 到 ``rank（input）`` 的维度执行规范化。默认1。
  - **epsilon** （float） - 添加到方差的很小的值，以防止除零。默认1e-05。
  - **param_attr** （ParamAttr | None） - 可学习增益g的参数属性。如果  ``scale`` 为False，则省略 ``param_attr`` 。如果 ``scale`` 为True且 ``param_attr`` 为None，则默认 ``ParamAttr`` 将作为比例。如果添加了 ``param_attr``， 则将其初始化为1。默认None。
  - **bias_attr** （ParamAttr | None） - 可学习偏差的参数属性b。如果 ``shift`` 为False，则省略 ``bias_attr`` 。如果 ``shift`` 为True且 ``param_attr`` 为None，则默认 ``ParamAttr`` 将作为偏差。如果添加了 ``bias_attr`` ，则将其初始化为0。默认None。
  - **act** （str） - 激活函数。默认 None
  - **name** （str） - 该层的名称， 可选的。默认为None，将自动生成唯一名称。

返回： 标准化后的结果   

**代码示例**

..  code-block:: python
    
   data = fluid.layers.data(name='data', shape=[3, 32, 32],
                                           dtype='float32')
   x = fluid.layers.layer_norm(input=data, begin_norm_axis=1)









.. _cn_api_fluid_layers_leaky_relu:

leaky_relu
-------------------------------

.. py:function:: paddle.fluid.layers.leaky_relu(x, alpha=0.02, name=None)

LeakyRelu 激活函数

.. math::   out=max(x,α∗x)

参数:
    - **x** (Variable) - LeakyRelu Operator的输入
    - **alpha** (FLOAT|0.02) - 负斜率，值很小。
    - **name** (str|None) - 此层的名称(可选)。如果设置为None，该层将被自动命名。

**代码示例：**

.. code-block:: python

    x = fluid.layers.data(name="x", shape=[2,3,16,16], dtype="float32")
    y = fluid.layers.leaky_relu(x, alpha=0.01)







.. _cn_api_fluid_layers_linear_chain_crf:

linear_chain_crf
-------------------------------

.. py:function:: paddle.fluid.layers.linear_chain_crf(input, label, param_attr=None)

线性链条件随机场（Linear Chain CRF）

条件随机场定义间接概率图，节点代表随机变量，边代表两个变量之间的依赖。CRF学习条件概率 :math:`P\left ( Y|X \right )` ， :math:`X = \left ( x_{1},x_{2},...,x_{n} \right )` 是结构性输入，:math:`Y = \left ( y_{1},y_{2},...,y_{n} \right )` 为输入标签。

线性链条件随机场（Linear Chain CRF)是特殊的条件随机场（CRF），有利于序列标注任务。序列标注任务不为输入设定许多条件依赖。唯一的限制是输入和输出必须是线性序列。因此类似CRF的图是一个简单的链或者线，也就是线性链随机场（linear chain CRF）。

该操作符实现了线性链条件随机场（linear chain CRF）的前向——反向算法。详情请参照 http://www.cs.columbia.edu/~mcollins/fb.pdf 和 http://cseweb.ucsd.edu/~elkan/250Bwinter2012/loglinearCRFs.pdf。

公式：

	1.这里x代表Emission

	2.Transition的第一维度值，代表起始权重，这里用a表示

	3.Transition的下一维值，代表末尾权重，这里用b表示

	4.Transition剩下的值，代表转移权重，这里用w表示

	5.Label用s表示

	长度为L的序列s的概率定义如下：

.. math::

    P(s) = (1/Z)exp(a_{s_{1}}+b_{s_{L}}+sum_{L}^{l=1}x_{s+{l}}+sum_{L}^{l=2}w_{s_{l-1},s_{l}})

其中Z是正则化值，所有可能序列的P(s)之和为1，x是线性链条件随机场（linear chain CRF）的发射（emission）特征权重。

线性链条件随机场最终输出mini-batch每个训练样本的条件概率的对数

**注意：**

    1.条件随机场（CRF）的特征函数由发射特征(emission feature）和转移特征（transition feature）组成。发射特征（emission feature）权重在调用函数前计算，而不在函数里计算。

    2.由于该函数对所有可能序列的进行全局正则化，发射特征（emission feature）权重应是未缩放的。因此如果该函数带有发射特征（emission feature），并且发射特征是任意非线性激活函数的输出，则请勿调用该函数。

    3.Emission的第二维度必须和标记数字（tag number）相同

参数：
    - **input** (Variable，LoDTensor，默认float类型LoDTensor) - 一个二维LoDTensor，shape为[N*D]，N是mini-batch的大小，D是总标记数。线性链条件随机场的未缩放发射权重矩阵
    - **input** (Tensor，默认float类型LoDTensor) - 一个二维张量，shape为[(D+2)*D]。linear_chain_crf操作符的可学习参数。更多详情见operator注释
    - **label** (Variable，LoDTensor，默认int64类型LoDTensor） - shape为[N*10的LoDTensor，N是mini-batch的总元素数
    - **param_attr** (ParamAttr) - 可学习参数的属性

返回：
    output(Variable，Tensor，默认float类型Tensor)：shape为[N*D]的二维张量。Emission的指数。这是前向计算中的中间计算结果，在后向计算中还会复用

    output(Variable，Tensor，默认float类型Tensor)：shape为[(D+2)*D]的二维张量。Transition的指数。这是前向计算中的中间计算结果，在后向计算中还会复用 

    output(Variable,Tensor，默认float类型Tensor)：mini-batch每个训练样本的条件概率的对数。这是一个shape为[S*1]的二维张量，S是mini-batch的序列数。注：S等同于mini-batch的序列数。输出不再是LoDTensor

返回类型：output（Variable）












.. _cn_api_fluid_layers_lod_reset:

lod_reset
-------------------------------

.. py:function:: paddle.fluid.layers.lod_reset(x, y=None, target_lod=None)


设定x的LoD为y或者target_lod。如果提供y，首先将y.lod指定为目标LoD,否则y.data将指定为目标LoD。如果未提供y，目标LoD则指定为target_lod。如果目标LoD指定为Y.data或target_lod，只提供一层LoD。

::


    * 例1:

    给定一级LoDTensor x:
        x.lod =  [[ 2,           3,                   1 ]]
        x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        x.dims = [6, 1]

    target_lod: [4, 2]

    得到一级LoDTensor:
        out.lod =  [[4,                          2]]
        out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        out.dims = [6, 1]

    * 例2:

    给定一级LoDTensor x:
        x.lod =  [[2,            3,                   1]]
        x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        x.dims = [6, 1]

    y是张量（Tensor）:
        y.data = [[2, 4]]
        y.dims = [1, 3]

    得到一级LoDTensor:
        out.lod =  [[2,            4]]
        out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        out.dims = [6, 1]

    * 例3:

    给定一级LoDTensor x:
        x.lod =  [[2,            3,                   1]]
        x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        x.dims = [6, 1]

    y是二级LoDTensor:
        y.lod =  [[2, 2], [2, 2, 1, 1]]
        y.data = [[1.1], [2.1], [3.1], [4.1], [5.1], [6.1]]
        y.dims = [6, 1]

    得到一个二级LoDTensor:
        out.lod =  [[2, 2], [2, 2, 1, 1]]
        out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        out.dims = [6, 1]

参数：
    - **x** (Variable)-输入变量，可以为Tensor或者LodTensor
    - **y** (Variable|None)-若提供，输出的LoD则衍生自y
    - **target_lod** (list|tuple|None)-一层LoD，y未提供时作为目标LoD

返回：输出变量，该层指定为LoD

返回类型：变量

提示：抛出异常 - 如果y和target_lod都为空

**代码示例**：

.. code-block:: python

    x = layers.data(name='x', shape=[10])
    y = layers.data(name='y', shape=[10, 20], lod_level=2)
    out = layers.lod_reset(x=x, y=y)









.. _cn_api_fluid_layers_log:

log
-------------------------------

.. py:function:: paddle.fluid.layers.log(x, name=None)


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











.. _cn_api_fluid_layers_log_loss:

log_loss
-------------------------------

.. py:function:: paddle.fluid.layers.log_loss(input, label, epsilon=0.0001, name=None)

**负log loss层**

该层对输入的预测结果和目的标签进行计算，返回负log loss损失值。

.. math::

    Out = -label * \log{(input + \epsilon)} - (1 - label) * \log{(1 - input + \epsilon)}


参数:
  - **input** (Variable|list) – 形为[N x 1]的二维张量, 其中N为batch大小。 该输入是由先前运算得来的概率集。
  - **label** (Variable|list) – 形为[N x 1]的二维张量，承载着正确标记的数据， 其中N为batch大小。
  - **epsilon** (float) – epsilon
  - **name** (string) – log_loss层的名称

返回：	形为[N x 1]的二维张量，承载着负log_loss值

返回类型:	变量（Variable）


**代码示例**

..  code-block:: python

	prob = fluid.layers.sigmoid(net)
        cost = fluid.layers.log_loss(input=prob, label=label)











.. _cn_api_fluid_layers_logical_and:

logical_and
-------------------------------

.. py:function:: paddle.fluid.layers.logical_and(x, y, out=None, name=None)

logical_and算子

它在X和Y上以元素方式操作，并返回Out。X、Y和Out是N维布尔张量（Tensor）。Out的每个元素的计算公式为：

.. math::
       Out = X \&\& Y

参数：
        - **x** （Variable）- （LoDTensor）logical_and运算符的左操作数
        - **y** （Variable）- （LoDTensor）logical_and运算符的右操作数
        - **out** （Tensor）- 输出逻辑运算的张量。
        - **name** （basestring | None）- 输出的名称。

返回：        (LoDTensor)n-dim bool张量。每个元素的计算公式： :math:`Out = X \&\& Y` 
        
返回类型：        输出（Variable）。        
        
        
**代码示例：**

.. code-block:: python

    left = fluid.layers.data(
        name='left', shape=[1], dtype='int32')
    right = fluid.layers.data(
        name='right', shape=[1], dtype='int32')
    result = fluid.layers.logical_and(x=left, y=right)









.. _cn_api_fluid_layers_logical_not:

logical_not
-------------------------------

.. py:function:: paddle.fluid.layers.logical_not(x, out=None, name=None)

logical_not算子

它在X上以元素方式操作，并返回Out。X和Out是N维布尔张量（Tensor）。Out的每个元素的计算公式为：

.. math:: 
        Out = !X

参数：
        - **x** （Variable）- （LoDTensor）logical_not运算符的操作数
        - **out** （Tensor）- 输出逻辑运算的张量。
        - **name** （basestring | None）- 输出的名称。

返回：        (LoDTensor)n维布尔张量。

返回类型：        输出（Variable）。        


**代码示例：**

.. code-block:: python
    
    left = fluid.layers.data(
        name='left', shape=[1], dtype='int32')
    result = fluid.layers.logical_not(x=left)







.. _cn_api_fluid_layers_logical_or:

logical_or
-------------------------------

.. py:function:: paddle.fluid.layers.logical_or(x, y, out=None, name=None)

logical_or算子

它在X和Y上以元素方式操作，并返回Out。X、Y和Out是N维布尔张量（Tensor）。Out的每个元素的计算公式为：

.. math:: 
        Out = X || Y

参数：
        - **x** （Variable）- （LoDTensor）logical_or运算符的左操作数
        - **y** （Variable）- （LoDTensor）logical_or运算符的右操作数
        - **out** （Tensor）- 输出逻辑运算的张量。
        - **name** （basestring | None）- 输出的名称。

返回：        (LoDTensor)n维布尔张量。每个元素的计算公式： :math:`Out = X || Y` 
        
返回类型：        输出（Variable）。        



**代码示例：**

.. code-block:: python

    left = fluid.layers.data(
        name='left', shape=[1], dtype='int32')
    right = fluid.layers.data(
        name='right', shape=[1], dtype='int32')
    result = fluid.layers.logical_or(x=left, y=right)






.. _cn_api_fluid_layers_logical_xor:

logical_xor
-------------------------------

.. py:function:: paddle.fluid.layers.logical_xor(x, y, out=None, name=None)

logical_xor算子

它在X和Y上以元素方式操作，并返回Out。X、Y和Out是N维布尔张量（Tensor）。Out的每个元素的计算公式为：

.. math:: 
        Out = (X || Y) \&\& !(X \&\& Y)

参数：
        - **x** （Variable）- （LoDTensor）logical_xor运算符的左操作数
        - **y** （Variable）- （LoDTensor）logical_xor运算符的右操作数
        - **out** （Tensor）- 输出逻辑运算的张量。
        - **name** （basestring | None）- 输出的名称。

返回：        (LoDTensor)n维布尔张量。
       
返回类型：        输出（Variable）。        



**代码示例：**

.. code-block:: python

    left = fluid.layers.data(
        name='left', shape=[1], dtype='int32')
    right = fluid.layers.data(
        name='right', shape=[1], dtype='int32')
    result = fluid.layers.logical_xor(x=left, y=right)






.. _cn_api_fluid_layers_lrn:

lrn
-------------------------------

.. py:function:: paddle.fluid.layers.lrn(input, n=5, k=1.0, alpha=0.0001, beta=0.75, name=None)

局部响应正则层（Local Response Normalization Layer）

该层对局部输入区域正则化，执行一种侧向抑制（lateral inhibition）。

公式如下：

.. math::

    Output(i,x,y) = Input(i,x,y)/\left ( k+\alpha \sum_{j=max(0,c-n/2)}^{min(C,c+n/2)}(Input(j,x,y))^2 \right )^\beta 

在以上公式中：
  - :math:`n` ：累加的通道数
  - :math:`k` ：位移（避免除数为0）
  - :math:`\alpha` ： 缩放参数
  - :math:`\beta` ： 指数参数

参考 : `ImageNet Classification with Deep Convolutional Neural Networks <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`_

参数：
    - **input** （Variable）- 该层输入张量，输入张量维度必须为4
    - **n** (int，默认5） - 累加哦的通道数
    - **k** （float，默认1.0）- 位移（通常为正数，避免除数为0）
    - **alpha** （float，默认1e-4）- 缩放参数
    - **beta** （float，默认0.75）- 指数
    - **name** （str，默认None）- 操作符名

抛出异常:
  - ``ValueError`` - 如果输入张量的阶不为4

返回：张量，存储转置结果

**代码示例**：

.. code-block:: python

    data = fluid.layers.data(
        name="data", shape=[3, 112, 112], dtype="float32")
    lrn = fluid.layers.lrn(input=data)











.. _cn_api_fluid_layers_lstm:

lstm
-------------------------------

.. py:function::  paddle.fluid.layers.lstm(input, init_h, init_c, max_len, hidden_size, num_layers, dropout_prob=0.0, is_bidirec=False, is_test=False, name=None, default_initializer=None, seed=-1)

如果您的设备是GPU，本op将使用cudnn LSTM实现

一个没有 peephole 连接的四门长短期记忆网络。在前向传播中，给定迭代的输出ht和单元输出ct可由递归输入ht-1、单元输入ct-1和上一层输入xt计算，给定矩阵W、R和bias bW, bR由下式计算:

.. math::

	i_t = \sigma(W_{ix}x_{t} + W_{ih}h_{t-1} + bx_i + bh_i)\\
	f_t = \sigma(W_{fx}x_{t} + W_{fh}h_{t-1} + bx_f + bh_f)\\
	o_t = \sigma(W_{ox}x_{t} + W_{oh}h_{t-1} + bx_o + bh_o)\\
	\tilde{c_t} = tanh(W_{cx}x_t + W_{ch}h_{t-1} + bx_c + bh_c)\\
	c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c_t}\\
	h_t = o_t \odot tanh(c_t)

公式中：
	- W 项表示权重矩阵(e.g. :math:`W_{ix}` 是从输入门到输入的权重矩阵)
	- b 项表示偏差向量( :math:`b_{xi}` 和 :math:`b_{hi}` 是输入门的偏差向量)
	- sigmoid 是 logistic sigmoid 函数
	- i、f、o、c 分别为输入门、遗忘门、输出门和激活向量，它们的大小与 cell 输出激活向量h相同。
	- :math:`\odot` 是向量的元素乘积
	- tanh是激活函数
	- :math:`\tilde{c_t}` 也称为候选隐藏状态，它是根据当前输入和之前的隐藏状态来计算的
	- sigmoid的计算公式为：sigmoid(x) = 1 / (1 + e^-x)
	- \*表示点乘，X表示矩阵乘

参数：
	- **input** (Variable) - LSTM 输入张量，形状必须为(seq_len x，batch_size，x，input_size)
	- **init_h** (Variable) – LSTM的初始隐藏状态，是一个有形状的张量(num_layers，x，batch_size，x，hidden_size)如果is_bidirec = True，形状应该是(num_layers*2，x， batch_size， x， hidden_size)
	- **init_c** (Variable) - LSTM的初始状态。这是一个有形状的张量(num_layers， x， batch_size， x， hidden_size)如果is_bidirec = True，形状应该是(num_layers*2， x， batch_size， x， hidden_size)
	- **max_len** (int) – LSTM的最大长度。输入张量的第一个 dim 不能大于max_len
	- **hidden_size** (int) - LSTM的隐藏大小
	- **num_layers** (int) –  LSTM的总层数
	- **dropout_prob** (float|0.0) – dropout prob，dropout 只在 rnn 层之间工作，而不是在时间步骤之间。dropout 不作用于最后的 rnn 层的 rnn 输出中
	- **is_bidirec** (bool) – 是否是双向的
	- **is_test** (bool) – 是否在测试阶段
	- **name** (str|None) - 此层的名称(可选)。如果没有设置，该层将被自动命名。
	- **default_initializer** (Initialize|None) – 在哪里使用初始化器初始化权重，如果没有设置，将进行默认初始化。
	- **seed** (int) – LSTM中dropout的Seed，如果是-1,dropout将使用随机Seed

返回：
	- LSTM hidden的结果：形状(seq_len x batch_size x hidden_size)如果is_bidirec设置为True,形状为(seq_len x batch_sze hidden_size * 2)
	- last_h(Tensor):  LSTM最后一步的隐藏状态，形状为(num_layers x batch_size x hidden_size)；如果is_bidirec设置为True，形状为(num_layers*2 x batch_size x hidden_size)
	- last_c(Tensor)： LSTM最后一步的cell状态，形状为(num_layers x batch_size x hidden_size)；如果is_bidirec设置为True，形状为(num_layers*2 x batch_size x hidden_size)

**代码示例：**

.. code-block:: python

	input = embedding
	batch_size = 20
	max_len = 100
	dropout_prob = 0.2
	input_size = 100
	hidden_size = 150
	num_layers = 1
	init_hidden1 = layers.fill_constant( [num_layers, batch_size, hidden_size], 'float32', 0.0, stop_grad=False)
	init_cell1 = layers.fill_constant( [num_layers, batch_size, hidden_size], 'float32', 0.0, stop_grad=False)

	rnn_out, last_h, last_c = layers.lstm( input, init_h, init_c, max_len, dropout_prob, input_size, hidden_size,  num_layers)












.. _cn_api_fluid_layers_lstm_unit:

lstm_unit
-------------------------------

.. py:function:: paddle.fluid.layers.lstm_unit(x_t, hidden_t_prev, cell_t_prev, forget_bias=0.0, param_attr=None, bias_attr=None, name=None)

Lstm unit layer

lstm步的等式：

.. math::

    i_{t} &= \sigma \left ( W_{x_{i}}x_{t}+W_{h_{i}}h_{t-1}+b_{i} \right ) \\
    f_{t} &= \sigma \left ( W_{x_{f}}x_{t}+W_{h_{f}}h_{t-1}+b_{f} \right ) \\
    c_{t} &= f_{t}c_{t-1}+i_{t}tanh\left ( W_{x_{c}}x_{t} +W_{h_{c}}h_{t-1}+b_{c}\right ) \\
    o_{t} &= \sigma \left ( W_{x_{o}}x_{t}+W_{h_{o}}h_{t-1}+b_{o} \right ) \\
    h_{t} &= o_{t}tanh \left ( c_{t} \right )

lstm单元的输入包括 :math:`x_{t}` ， :math:`h_{t-1}` 和 :math:`c_{t-1}` 。:math:`h_{t-1}` 和 :math:`c_{t-1}` 的第二维应当相同。在此实现过程中，线性转换和非线性转换分离。以 :math:`i_{t}` 为例。线性转换运用到fc层，等式为：

.. math::

    L_{i_{t}} = W_{x_{i}}x_{t} + W_{h_{i}}h_{t-1} + b_{i}

非线性转换运用到lstm_unit运算，方程如下：

.. math::

    i_{t} = \sigma \left ( L_{i_{t}} \right )

该层有 :math:`h_{t}` 和 :math:`o_{t}` 两个输出。

参数：
    - **x_t** (Variable) - 当前步的输入值，二维张量，shape为 M x N ，M是批尺寸，N是输入尺寸
    - **hidden_t_prev** (Variable) - lstm单元的隐藏状态值，二维张量，shape为 M x S，M是批尺寸，N是lstm单元的大小
    - **cell_t_prev** (Variable) - lstm单元的cell值，二维张量，shape为 M x S ，M是批尺寸，N是lstm单元的大小
    - **forget_bias** (Variable) - lstm单元的遗忘bias
    - **param_attr** (ParamAttr|None) - 可学习hidden-hidden权重的擦参数属性。如果设为None或者ParamAttr的一个属性，lstm_unit创建ParamAttr为param_attr。如果param_attr的初始化函数未设置，参数初始化为Xavier。默认：None
    - **bias_attr** (ParamAttr|None) - 可学习bias权重的bias属性。如果设为False，输出单元中则不添加bias。如果设为None或者ParamAttr的一个属性，lstm_unit创建ParamAttr为bias_attr。如果bias_attr的初始化函数未设置，bias初始化为0.默认：None
    - **name** (str|None) - 该层名称（可选）。若设为None，则自动为该层命名

返回：lstm单元的hidden(隐藏状态)值和cell值

返回类型：tuple（元组）

抛出异常:
  - ``ValueError`` - ``x_t``，``hidden_t_prev`` 和 ``cell_t_prev`` 的阶不为2，或者 ``x_t`` ，``hidden_t_prev`` 和 ``cell_t_prev`` 的第一维不一致，或者 ``hidden_t_prev`` 和 ``cell_t_prev`` 的第二维不一致

**代码示例**：

.. code-block:: python

    x_t = fluid.layers.fc(input=x_t_data, size=10)
    prev_hidden = fluid.layers.fc(input=prev_hidden_data, size=30)
    prev_cell = fluid.layers.fc(input=prev_cell_data, size=30)
    hidden_value, cell_value = fluid.layers.lstm_unit(x_t=x_t,
                                       hidden_t_prev=prev_hidden,
                                       cell_t_prev=prev_cell)











.. _cn_api_fluid_layers_margin_rank_loss:

margin_rank_loss
-------------------------------

.. py:function:: paddle.fluid.layers.margin_rank_loss(label, left, right, margin=0.1, name=None)

margin rank loss（差距排序损失）层。在排序问题中，它可以比较传进来的 ``left`` 得分和 ``right`` 得分。

可用如下等式定义：

.. math::
    rank\_loss = max(0, -label * (left - right) + margin)


参数:
  - **label** (Variable) – 表明是否左元素排名要高于右元素
  - **left** (Variable) – 左元素排序得分
  - **right** (Variable) – 右元素排序得分
  - **margin** (float) – 指定固定的得分差
  - **name** (str|None) – 可选项，该层的命名。如果为None, 该层将会被自动命名

返回： 排序损失

返回类型:	变量（Variable）

抛出异常: 
  - ``ValueError`` - ``label`` , ``left`` , ``right`` 有一者不为Variable类型时，抛出此异常
 
**代码示例**

..  code-block:: python

             label = fluid.layers.data(name="label", shape=[4, 1], dtype="float32")
             left = fluid.layers.data(name="left", shape=[4, 1], dtype="float32")
             right = fluid.layers.data(name="right", shape=[4, 1], dtype="float32")
             out = fluid.layers.margin_rank_loss(label, left, right)











.. _cn_api_fluid_layers_matmul:



matmul
-------------------------------

.. py:function:: paddle.fluid.layers.matmul(x, y, transpose_x=False, transpose_y=False, alpha=1.0, name=None)

对两个张量进行矩阵相乘

当前输入的张量可以为任意阶，但当任意一个输入的阶数大于3时，两个输入的阶必须相等。
实际的操作取决于x,y的维度和 ``transpose_x`` , ``transpose_y`` 的标记值。具体如下：

- 如果transpose值为真，则对应 ``tensor`` 的最后两位将被转置。如：x是一个shape=[D]的一阶张量，那么x在非转置形式中为[1,D]，在转置形式中为[D,1],而y则相反，在非转置形式中作为[D,1]，在转置形式中作为[1,D]。

- 转置后，这两个`tensors`将为 2-D 或 n-D ,并依据下列规则进行矩阵相乘：
	- 如果两个都是2-D，则同普通矩阵一样进行矩阵相乘
	- 如果任意一个是n-D，则将其视为驻留在最后两个维度的矩阵堆栈，并在两个张量上应用支持广播的批处理矩阵乘法。

**注意，如果原始张量x或y的秩为1且没有转置，则在矩阵乘法之后，前置或附加维度1将被移除。**


参数：
    - **x** (Variable)-输入变量，类型为Tensor或LoDTensor
    - **y** (Variable)-输入变量，类型为Tensor或LoDTensor
    - **transpose_x** (bool)-相乘前是否转置x
    - **transeptse_y** (bool)-相乘前是否转置y
    - **alpha** (float)-输出比例。默认为1.0
    - **name** (str|None)-该层名称（可选）。如果设置为空，则自动为该层命名

返回：张量乘积变量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    # 以下是解释输入和输出维度的示例
    # x: [B, ..., M, K], y: [B, ..., K, N]
    fluid.layers.matmul(x, y)  # out: [B, ..., M, N]

    # x: [B, M, K], y: [B, K, N]
    fluid.layers.matmul(x, y)  # out: [B, M, N]

    # x: [B, M, K], y: [K, N]
    fluid.layers.matmul(x, y)  # out: [B, M, N]

    # x: [M, K], y: [K, N]
    fluid.layers.matmul(x, y)  # out: [M, N]

    # x: [B, M, K], y: [K]
    fluid.layers.matmul(x, y)  # out: [B, M]

    # x: [K], y: [K]
    fluid.layers.matmul(x, y)  # out: [1]

    # x: [M], y: [N]
    fluid.layers.matmul(x, y, True, True)  # out: [M, N]









.. _cn_api_fluid_layers_maxout:

maxout
-------------------------------

.. py:function:: paddle.fluid.layers.maxout(x, groups, name=None)

假设输入形状为(N, Ci, H, W)，输出形状为(N, Co, H, W)，则 :math:`Co=Ci/groups` 运算公式如下:

.. math:: 

	y_{si+j} &= \max_k x_{gsi + sk + j} \\
	g &= groups \\
	s &= \frac{input.size}{num_channels} \\
	0 &\le i < \frac{num_channels}{groups} \\
	0 &\le j < s \\
	0 &\le k < groups


请参阅论文:
	- Maxout Networks:  http://www.jmlr.org/proceedings/papers/v28/goodfellow13.pdf
	- Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks: https://arxiv.org/pdf/1312.6082v4.pdf

参数：
	- **x** (Variable) - (tensor) maxout算子的输入张量。输入张量的格式为NCHW。其中N为 batch size ，C为通道数，H和W为feature的高和宽
	- **groups** （INT）- 指定输入张量将被分成多少组“通道维数”。输出通道的数量以组为单位。
	- **name** (basestring|None) - 输出的名称

返回：Tensor，maxout算子的输出张量。输出张量的格式也是NCHW。其中N为 batch size，C为通道数，H和W为特征的高和宽。

返回类型：out（variable）










.. _cn_api_fluid_layers_mean:

mean
-------------------------------

.. py:function:: paddle.fluid.layers.mean(x, name=None)
       
mean算子计算X中所有元素的平均值
     
参数：
        - **x** (Variable)- (Tensor) 均值运算的输入。
        - **name** (basestring | None)- 输出的名称。

返回：       均值运算输出张量（Tensor）
       
返回类型：        Variable
        
        








.. _cn_api_fluid_layers_mean_iou:

mean_iou
-------------------------------

.. py:function:: paddle.fluid.layers.mean_iou(input, label, num_functiones)

均值IOU（Mean  Intersection-Over-Union）是语义图像分割中的常用的评价指标之一，它首先计算每个语义类的IOU，然后计算类之间的平均值。定义如下:
      
          .. math::   IOU = \frac{true_positi}{true_positive+false_positive+false_negative}
          
在一个混淆矩阵中累积得到预测值，然后从中计算均值-IOU。

参数:
    - **input** (Variable) - 类型为int32或int64的语义标签的预测结果张量。
    - **label** (Variable) - int32或int64类型的真实label张量。它的shape应该与输入相同。
    - **num_classes** (int) - 标签可能的类别数目。
    
返回: 张量，shape为[1]， 代表均值IOU。out_wrong(变量):张量，shape为[num_classes]。每个类别中错误的个数。out_correct(变量):张量，shape为[num_classes]。每个类别中的正确个数。

返回类型: mean_iou(Variable)

**代码示例**:

..  code-block:: python

   iou, wrongs, corrects = fluid.layers.mean_iou(predict, label, num_classes)









.. _cn_api_fluid_layers_merge_selected_rows:

merge_selected_rows
-------------------------------

.. py:function:: paddle.fluid.layers.merge_selected_rows(x, name=None)

**实现合并选中行（row）操作**

该运算用于合并（值相加）输入张量中重复的行。输出行没有重复的行，并且按值从小到大顺序重新对行排序。

::
    
    例如：
          
          输入: 
               X.rows = [0, 5, 5, 4, 19]
               X.height = 20
               X.value = [[1, 1] [2, 2] [3, 3] [4, 4] [6, 6]]
          
          
          输出：
               Out.row is [0, 4, 5, 19] 
               Out.height is 20 
               Out.value is: [[1, 1] [4, 4] [5, 5] [6, 6]]



参数:
  - x (Variable) – 输入类型为SelectedRows, 选中行有可能重复
  - name (basestring|None) – 输出变量的命名

返回: 输出类型为SelectedRows，并且选中行不会重复 

返回类型:	变量（Variable）











.. _cn_api_fluid_layers_mul:

mul
-------------------------------

.. py:function:: paddle.fluid.layers.mul(x, y, x_num_col_dims=1, y_num_col_dims=1, name=None)
        
mul算子
此运算是用于对输入X和Y执行矩阵乘法。
等式是：

.. math:: 
        Out = X * Y

输入X和Y都可以携带LoD（详细程度）信息。但输出仅与输入X共享LoD信息。

参数：
        - **x** (Variable)- (Tensor) 乘法运算的第一个输入张量。
        - **y** (Variable)- (Tensor) 乘法运算的第二个输入张量。
        - **x_num_col_dims** （int）- 默认值1， 可以将具有两个以上维度的张量作为输入。如果输入X是具有多于两个维度的张量，则输入X将先展平为二维矩阵。展平规则是：前 ``num_col_dims`` 将被展平成最终矩阵的第一个维度（矩阵的高度），其余的 rank(X) - num_col_dims 维度被展平成最终矩阵的第二个维度（矩阵的宽度）。结果是展平矩阵的高度等于X的前 ``x_num_col_dims`` 维数的乘积，展平矩阵的宽度等于X的最后一个秩（x）- ``num_col_dims`` 个剩余维度的维数的乘积。例如，假设X是一个五维张量，形状为（2,3,4,5,6）。 则扁平化后的张量具有的形即为 （2x3x4,5x6）=（24,30）。
        - **y_num_col_dims** （int）- 默认值1， 可以将具有两个以上维度的张量作为输入。如果输入Y是具有多于两个维度的张量，则Y将首先展平为二维矩阵。 ``y_num_col_dims`` 属性确定Y的展平方式。有关更多详细信息，请参阅 ``x_num_col_dims`` 的注释。
        - **name** (basestring | None)- 输出的名称。

返回：       乘法运算输出张量（Tensor）.
       
返回类型：    输出(Variable)。       
        
        








.. _cn_api_fluid_layers_multiplex:

multiplex
-------------------------------

.. py:function:: paddle.fluid.layers.multiplex(inputs, index)

引用给定的索引变量，该层从输入变量中选择行构造Multiplex变量。 

假设有 :math:`m` 个输入变量，:math:`I_{i}` 代表第i个输入变量，而且 :math:`i` is in :math:`[0,m)` 。 

所有输入变量都是具有相同形状的张量 :math:`[d_0,d_1, ... ,d_R]` 。 

请注意，输入张量的秩应至少为2。每个输入变量将被视为形状为 :math:`[M，N]` 的二维矩阵，其中 :math:`M` 表示 :math:`d0` ，N表示 :math:`d_1 * d_2 * ... * d_R` 。 

设 :math:`I_{i}[j]` 为第i个输入变量的第j行。 给定的索引变量是具有形状[M，1]的2-D张量。 设 :math:`ID[i]` 为索引变量的第i个索引值。 然后输出变量将是一个形状为 :math:`[d_0,d_1, ... ,d_R]` 的张量。 

如果将输出张量视为具有形状[M，N]的2-D矩阵,并且令O[i]为矩阵的第i行，则O[i]等于 :math:`I_{ID}[i][i]` 
  
- Ids: 索引张量
- X[0 : N - 1]: 输出的候选张量度(N >= 2).
- 对于从 0 到 batchSize-1 的每个索引i，输出是第（Ids [i]）  张量的第i行

对于第i行的输出张量：

.. math::
            \\y[i]=x_k[i]\\

其中 :math:`y` 为输出张量， :math:`x_k` 为第k个输入张量，并且 :math:`k=Ids[i]` 。

参数:
  - **inputs** （list） - 要从中收集的变量列表。所有变量的形状相同，秩至少为2
  - **index** （Variable） -  Tensor <int32>，索引变量为二维张量，形状[M, 1]，其中M为批大小。

返回：multiplex 张量

**代码示例**

..  code-block:: python

   import paddle.fluid as fluid
   
   x1 = fluid.layers.data(name='x1', shape=[4], dtype='float32')
   x2 = fluid.layers.data(name='x2', shape=[4], dtype='float32')
   index = fluid.layers.data(name='index', shape=[1], dtype='int32')
   out = fluid.layers.multiplex(inputs=[x1, x2], index=index)
   









.. _cn_api_fluid_layers_nce:

nce
-------------------------------

.. py:function:: paddle.fluid.layers.nce(input, label, num_total_functiones, sample_weight=None, param_attr=None, bias_attr=None, num_neg_samples=None, name=None, sampler='uniform', custom_dist=None, seed=0, is_sparse=False)

计算并返回噪音对比估计（ noise-contrastive estimation training loss）。
`请参考 See Noise-contrastive estimation: A new estimation principle for unnormalized statistical models 
<http://www.jmlr.org/proceedings/papers/v9/gutmann10a/gutmann10a.pdf>`_ 
该operator默认使用均匀分布进行抽样。

参数:
    - **input** (Variable) -  输入变量
    - **label** (Variable) -  标签
    - **num_total_classes** (int) - 所有样本中的类别的总数
    - **sample_weight** (Variable|None) - 存储每个样本权重，shape为[batch_size, 1]存储每个样本的权重。每个样本的默认权重为1.0
    - **param_attr** (ParamAttr|None) - :math:`可学习参数/nce权重` 的参数属性。如果它没有被设置为ParamAttr的一个属性，nce将创建ParamAttr为param_attr。如没有设置param_attr的初始化器，那么参数将用Xavier初始化。默认值:None
    - **bias_attr** (ParamAttr|bool|None) -  nce偏置的参数属性。如果设置为False，则不会向输出添加偏置（bias）。如果值为None或ParamAttr的一个属性，则bias_attr=ParamAtt。如果没有设置bias_attr的初始化器，偏置将被初始化为零。默认值:None
    - **num_neg_samples** (int) - 负样例的数量。默认值是10
    - **name** (str|None) - 该layer的名称(可选)。如果设置为None，该层将被自动命名
    - **sampler** (str) – 取样器，用于从负类别中进行取样。可以是 ‘uniform’, ‘log_uniform’ 或 ‘custom_dist’。 默认 ‘uniform’
    - **custom_dist** (float[]) – 一个 float[] 并且它的长度为 ``num_total_classes`` 。  如果取样器类别为‘custom_dist’，则使用此参数。 custom_dist[i] 是第i个类别被取样的概率。默认为 None
    - **seed** (int) – 取样器使用的seed。默认为0
    - **is_sparse** (bool) – 标志位，指明是否使用稀疏更新,  :math:`weight@GRAD` 和 :math:`bias@GRAD` 会变为 SelectedRows

返回：	nce loss

返回类型:	变量（Variable）


**代码示例**

..  code-block:: python

		window_size = 5
		words = []
		for i in xrange(window_size):
			words.append(layers.data(
				name='word_{0}'.format(i), shape=[1], dtype='int64'))

		dict_size = 10000
		label_word = int(window_size / 2) + 1

		embs = []
		for i in xrange(window_size):
			if i == label_word:
				continue

			emb = layers.embedding(input=words[i], size=[dict_size, 32],
								   param_attr='emb.w', is_sparse=True)
			embs.append(emb)

		embs = layers.concat(input=embs, axis=1)
		loss = layers.nce(input=embs, label=words[label_word],
					  num_total_classes=dict_size, param_attr='nce.w',
					  bias_attr='nce.b')

		#使用custom distribution
		dist = fluid.layers.assign(input=np.array([0.05,0.5,0.1,0.3,0.05]).astype("float32"))
		loss = layers.nce(input=embs, label=words[label_word],
					  num_total_classes=5, param_attr='nce.w',
					  bias_attr='nce.b',
					  num_neg_samples=3,
					  sampler="custom_dist",
					  custom_dist=dist)










.. _cn_api_fluid_layers_one_hot:

one_hot 
-------------------------------

.. py:function:: paddle.fluid.layers.one_hot(input, depth)

该层创建输入指数的one-hot表示

参数：
    - **input** (Variable)-输入指数，最后维度必须为1
    - **depth** (scalar)-整数，定义one-hot维度的深度

返回：输入的one-hot表示

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python 

    label = layers.data(name="label", shape=[1], dtype="float32")
    one_hot_label = layers.one_hot(input=label, depth=10)









.. _cn_api_fluid_layers_pad:

pad
-------------------------------

.. py:function:: paddle.fluid.layers.pad(x, paddings, pad_value=0.0, name=None)

在张量上加上一个由 ``pad_value`` 给出的常数值，填充宽度由 ``paddings`` 指定。
其中，维度 ``i`` 中 ``x`` 内容前填充的值个数用 ``paddings[i]`` 表示，维度 ``i`` 中 ``x`` 内容后填充的值个数用 ``paddings[i+1]`` 表示。
   
一个例子:

::

        Given:

         x = [[1, 2], [3, 4]]

        paddings = [0, 1, 1, 2]

        pad_value = 0

        Return:

        out = [[0, 1, 2, 0, 0]
               [0, 3, 4, 0, 0]
               [0, 0, 0, 0, 0]]


参数:
    - **x** (Variable) — —输入张量变量。
    - **paddings** (list) — 一个整数列表。按顺序填充在每个维度上填充元素。 ``padding`` 长度必须是 ``rank(x)×2``
    - **pad_value** (float) — 用来填充的常量值。
    - **name** (str|None) — 这个层的名称(可选)。如果设置为None，该层将被自动命名。

返回：	填充后的张量变量

返回类型： 变量（Variable）
    

**代码示例**

..  code-block:: python
        
    out = fluid.layers.pad(
    x=x, paddings=[0, 1, 1, 2], pad_value=0.)










.. _cn_api_fluid_layers_pad2d:

pad2d
-------------------------------

.. py:function::  paddle.fluid.layers.pad2d(input, paddings=[0, 0, 0, 0], mode='constant', pad_value=0.0, data_format='NCHW', name=None)

依照 paddings 和 mode 属性对图像进行2维 ``pad``,如果mode是 ``reflection``，则paddings[0]和paddings[1]必须不大于height-1。宽度维数具有相同的条件。

例如：

.. code-block:: text

	假设X是输入图像:

      X = [[1, 2, 3],
           [4, 5, 6]]
     
     Case 0:
        paddings = [0, 1, 2, 3],
        mode = 'constant'
        pad_value = 0
        Out = [[0, 0, 1, 2, 3, 0, 0, 0]
               [0, 0, 4, 5, 6, 0, 0, 0]
               [0, 0, 0, 0, 0, 0, 0, 0]]
     
     Case 1:
        paddings = [0, 1, 2, 1],
        mode = 'reflect'
        Out = [[3, 2, 1, 2, 3, 2]
               [6, 5, 4, 5, 6, 5]
               [3, 2, 1, 2, 3, 2]]
     
     Case 2:
        paddings = [0, 1, 2, 1],
        mode = 'edge'
        Out = [[1, 1, 1, 2, 3, 3]
               [4, 4, 4, 5, 6, 6]
               [4, 4, 4, 5, 6, 6]]

参数：
	- **input** (Variable) - 具有[N, C, H, W]格式或[N, H, W, C]格式的输入图像。
	- **paddings** (tuple|list) - 填充区域的大小。如果填充是一个元组，它必须包含四个整数，
	  (padding_top、padding_bottom、padding_left、padding_right)。默认:padding =[0,0,0,0]。
	- **mode** (str) - 三种模式:constant(默认)、reflect、edge。默认值:常数
	- **pad_value** (float32) - 以常量模式填充填充区域的值。默认值:0
	- **data_format** (str)  - 可选字符串，选项有: ``NHWC`` , ``NCHW``。指定输入数据的数据格式。默认值:``NCHW``
	- **name** (str|None) - 此层的名称(可选)。如果没有设置，该层将被自动命名。

返回： tensor变量，按照 padding值 和 mode 进行填充

返回类型：variable

**代码示例：**

.. code-block:: python

	data = fluid.layers.data(name='data', shape=[3, 32, 32], dtype='float32')
	result = fluid.layers.pad2d(input=data, padding=[1,2,3,4], mode='reflect')











.. _cn_api_fluid_layers_pad_constant_like:

pad_constant_like
-------------------------------

.. py:function:: paddle.fluid.layers.pad_constant_like(x, y, pad_value=0.0, name=None)

使用 ``pad_value`` 填充 ``Y`` ，填充到每个axis（轴）值的数量由X和Y的形不同而指定。（（0，shape_x_0 - shape_y_0），...（0，shape_x_n - shape_y_n ））是每个axis唯一pad宽度。输入应该是k维张量（k> 0且k <7）。

**实例如下**

::

    Given:
        X = [[[[ 0,  1,  2],
               [ 3,  4,  5]],
              [[ 6,  7,  8],
               [ 9, 10, 11]],
              [[12, 13, 14],
               [15, 16, 17]]],
             [[[18, 19, 20],
               [21, 22, 23]],
              [[24, 25, 26],
               [27, 28, 29]],
              [[30, 31, 32],
               [33, 34, 35]]]]
        X.shape = (2, 3, 2, 3)

        Y = [[[[35, 36, 37]],
              [[38, 39, 40]],
              [[41, 42, 43]]]]
        Y.shape = (1, 3, 1, 3)
        
参数：
          - **x** （Variable）- 输入Tensor变量。
          - **y** （Variable）- 输出Tensor变量。
          - **pad_value** (float) - 用于填充的常量值。
          - **name** （str | None） - 这一层的名称（可选）。如果设置为None，则将自动命名这一层。

返回：填充张量（Tensor）变量

返回类型：  变量（Variable）

**示例代码**

..  code-block:: python

    # x是秩为4的tensor, x.shape = (2, 3, 2, 3)。
    # y是秩为4的tensor, y.shape = (1, 3, 1, 3)。
    out = fluid.layers.pad_constant_like(x=x, y=y, pad_value=0.)
    # out是秩为4的tensor, out.shape = [2, 3 ,2 , 3]。











.. _cn_api_fluid_layers_pool2d:

pool2d
-------------------------------

.. py:function:: paddle.fluid.layers.pool2d(input, pool_size=-1, pool_type='max', pool_stride=1, pool_padding=0, global_pooling=False, use_cudnn=True, ceil_mode=False, name=None, exclusive=True)

pooling2d操作符根据 ``input`` ， 池化类型 ``pool_type`` ， 池化核大小 ``pool_size`` , 步长 ``pool_stride`` ，填充 ``pool_padding`` 这些参数得到输出。

输入X和输出Out是NCHW格式，N为batch尺寸，C是通道数，H是特征高度，W是特征宽度。

参数（ksize,strides,paddings）含有两个元素。这两个元素分别代表高度和宽度。输入X的大小和输出Out的大小可能不一致。

例如：

输入：
    X shape：:math:`\left ( N,C,H_{in},W_{in} \right )`

输出：
    Out shape：:math:`\left ( N,C,H_{out},W_{out} \right )`

如果 ``ceil_mode`` = false：

.. math::
    H_{out} = \frac{(H_{in} - ksize[0] + 2 * paddings[0])}{strides[0]} + 1

.. math::
    W_{out} = \frac{(W_{in} - ksize[1] + 2 * paddings[1])}{strides[1]} + 1

如果 ``ceil_mode`` = true：

.. math::
    H_{out} = \frac{(H_{in} - ksize[0] + 2 * paddings[0] + strides[0] - 1)}{strides[0]} + 1

.. math::
    W_{out} = \frac{(W_{in} - ksize[1] + 2 * paddings[1] + strides[1] - 1)}{strides[1]} + 1

如果 ``exclusive`` = true:

.. math::
    hstart &= i * strides[0] - paddings[0] \\
    hend   &= hstart + ksize[0] \\
    wstart &= j * strides[1] - paddings[1] \\
    wend   &= wstart + ksize[1] \\
    Output(i ,j) &= \frac{sum(Input[hstart:hend, wstart:wend])}{ksize[0] * ksize[1]}

如果 ``exclusive`` = false:

.. math::
    hstart &= max(0, i * strides[0] - paddings[0])\\
    hend &= min(H, hstart + ksize[0]) \\
    wstart &= max(0, j * strides[1] - paddings[1]) \\
    wend & = min(W, wstart + ksize[1]) \\
    Output(i ,j) & = \frac{sum(Input[hstart:hend, wstart:wend])}{(hend - hstart) * (wend - wstart)}



参数：
    - **input** (Variable) - 池化操作的输入张量。输入张量格式为NCHW，N为批尺寸，C是通道数，H是特征高度，W是特征宽度
    - **pool_size** (int|list|tuple)  - 池化窗口的大小。如果它是一个元组或列表，它必须包含两个整数值， (pool_size_Height, pool_size_Width)。否则它应是一个整数的平方值。
    - **pool_type** (string) - 池化类型，可以是“max”对应max-pooling，“avg”对应average-pooling
    - **pool_stride** (int|list|tuple)  - 池化层的步长。如果它是一个元组或列表，它将包含两个整数，(pool_stride_Height, pool_stride_Width)。否则它是一个整数的平方值。
    - **pool_padding** (int|list|tuple) - 填充大小。如果它是一个元组或列表，它必须包含两个整数值，(pool_padding_on_Height, pool_padding_on_Width)。否则它是一个整数的平方值。
    - **global_pooling** （bool，默认false）- 是否用全局池化。如果global_pooling = true， ``pool_size`` 和 ``pool_padding`` 将被忽略。
    - **use_cudnn** （bool，默认false）- 只在cudnn核中用，需要下载cudnn
    - **ceil_mode** （bool，默认false）- 是否用ceil函数计算输出高度和宽度。默认False。如果设为False，则使用floor函数
    - **name** （str|None） - 该层名称（可选）。若设为None，则自动为该层命名。
    - **exclusive** (bool) - 是否在平均池化模式忽略填充值。默认为True。

返回：池化结果

返回类型：变量（Variable）

抛出异常：
    - ``ValueError`` - 如果 ``pool_type`` 既不是“max”也不是“avg”
    - ``ValueError`` - 如果 ``global_pooling`` 为False并且‘pool_size’为-1
    - ``ValueError`` - 如果 ``use_cudnn`` 不是bool值

**代码示例**：

.. code-block:: python

    data = fluid.layers.data(
        name='data', shape=[3, 32, 32], dtype='float32')
    conv2d = fluid.layers.pool2d(
                  input=data,
                  pool_size=2,
                  pool_type='max',
                  pool_stride=1,
                  global_pooling=False)









.. _cn_api_fluid_layers_pool3d:

pool3d
-------------------------------

.. py:function:: paddle.fluid.layers.pool3d(input, pool_size=-1, pool_type='max', pool_stride=1, pool_padding=0, global_pooling=False, use_cudnn=True, ceil_mode=False, name=None， exclusive=True)

函数使用上述输入参数的池化配置，为三维空间添加池化操作

参数：
    - **input** (Vairable) - 池化运算的输入张量。输入张量的格式为NCDHW, N是批尺寸，C是通道数，D是特征深度，H是特征高度，W是特征宽度。
    - **pool_size** (int|list|tuple) - 池化窗口的大小。如果为元组类型，那么它应该是由三个整数组成：深度，高度，宽度。如果是int类型，它应该是一个整数的立方。
    - **pool_type** (str) - 池化类型， "max" 对应max-pooling, "avg" 对应average-pooling。
    - **pool_stride** (int|list|tuple) - 池化跨越步长。如果为元组类型，那么它应该是由三个整数组成：深度，高度，宽度。如果是int类型，它应该是一个整数的立方。
    - **pool_padding** (int|list|tuple) - 填充大小。如果为元组类型，那么它应该是由三个整数组成：深度，高度，宽度。如果是int类型，它应该是一个整数的立方。
    - **global_pooling** (bool) - 是否使用全局池化。如果global_pooling = true, ``pool_size`` 和 ``pool_padding`` 将被忽略。
    - **use_cudnn** (bool) - 是否用cudnn核，只有在cudnn库安装时有效。
    - **ceil_mode** (bool) - 是否用ceil函数计算输出高度和宽度。默认False。如果设为False，则使用floor函数。
    - **name** (str) - 该层名称（可选）。若为空，则自动为该层命名。
    - **exclusive** (bool) - 是否在平均池化模式忽略填充值。默认为True。

返回：pool3d层的输出

返回类型：变量（Variable）












.. _cn_api_fluid_layers_pow:

pow
-------------------------------

.. py:function:: paddle.fluid.layers.pow(x, factor=1.0, name=None)

指数激活算子（Pow Activation Operator.）

参数
    - **x** (Variable) - Pow operator的输入
    - **factor** (FLOAT|1.0) - Pow的指数因子
    - **name** (str|None) -这个层的名称(可选)。如果设置为None，该层将被自动命名。

返回: 输出Pow操作符

返回类型: 输出(Variable)


**代码示例：**

.. code-block:: python

    x = fluid.layers.data(name="x", shape=[3,10,32,32], dtype="float32")
    y = fluid.layers.pow(x, factor=2.0)






.. _cn_api_fluid_layers_prelu:

prelu
-------------------------------

.. py:function:: paddle.fluid.layers.prelu(x, mode, param_attr=None, name=None)

等式：

.. math::
    y = max(0, x) + \alpha min(0, x)

参数：
          - **x** （Variable）- 输入为Tensor。
          - **param_attr** (ParamAttr|None) - 可学习权重 :math:`[\alpha]` 的参数属性。
          - **mode** （string）- 权重共享的模式all：所有元素共享相同的权重通道：通道中的元素共享相同的权重元素：每个元素都有一个权重
          - **name** （str | None）- 这一层的名称（可选）。如果设置为None，则将自动命名这一层。

返回： 输出Tensor与输入shape相同。

返回类型：  变量（Variable）
  
  
  








.. _cn_api_fluid_layers_random_crop:

random_crop
-------------------------------

.. py:function:: paddle.fluid.layers.random_crop(x, shape, seed=None)

该operator对batch中每个实例进行随机裁剪。这意味着每个实例的裁剪位置不同，裁剪位置由均匀分布随机生成器决定。所有裁剪的实例都具有相同的shape，由参数shape决定。

参数:
    - **x(Variable)** - 一组随机裁剪的实例
    - **shape(int)** - 裁剪实例的形状
    - **seed(int|变量|None)** - 默认情况下，随机种子从randint(-65536,-65536)中取得

返回: 裁剪后的batch

**代码示例**:

..  code-block:: python

   img = fluid.layers.data("img", [3, 256, 256])
   cropped_img = fluid.layers.random_crop(img, shape=[3, 224, 224])










.. _cn_api_fluid_layers_rank_loss:

rank_loss
-------------------------------


.. py:function::  paddle.fluid.layers.rank_loss(label, left, right, name=None)

`RankNet <http://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf>`_ 是一个成对的
排序模型，训练样本由一对文档组成：A和B。标签P表示a的排名是否高于B:

P 的取值可为： {0, 1} 或 {0, 0.5, 1}, 其中，0.5表示输入的两文档排序相同。

排序的损失函数有三个输入:left(o_i)、right(o_j) 和 label (P\_{i,j})。输入分别表示RankNet对文档A、B的输出得分和标签p的值。由下式计算输入的排序损失C\_{i,j}:

.. math::

	 C_{i,j} &= -\tilde{P_{ij}} * o_{i,j} + \log(1 + e^{o_{i,j}}) \\
      o_{i,j} &=  o_i - o_j  \\
      \tilde{P_{i,j}} &= \left \{0, 0.5, 1 \right \} \ or \ \left \{0, 1 \right \}

排序损失层的输入为batch_size (batch_size >= 1)

参数：
	- **label** (Variable)：A的排名是否高于B
	- **left** (Variable)：RankNet对doc A的输出分数
	- **right** (Variable): RankNet对doc B的输出分数
	- **name** (str|None)：此层的名称(可选)。如果没有设置，层将自动命名。

返回：variable：形为[N, H, W, 2]的输出

抛出异常：ValueError: 如果参数为不支持的类型


**代码示例**

.. code-block:: python

    theta = fluid.layers.data(name="x", shape=[2, 3], dtype="float32")
    out_shape = fluid.layers.data(name="y", shape=[-1], dtype="float32")
    data = fluid.layers.affine_grid(theta, out_shape)
    
    # or
    data = fluid.layers.affine_grid(theta, [5, 3, 28, 28])










.. _cn_api_fluid_layers_reduce_max:

reduce_max
-------------------------------

.. py:function:: paddle.fluid.layers.reduce_max(input, dim=None, keep_dim=False, name=None)

计算给定维度上张量（Tensor）元素最大值。

参数：
          - **input** （Variable）：输入变量为Tensor或LoDTensor。
          - **dim** （list | int | None）：函数运算的维度。如果为None，则计算所有元素的平均值并返回单个元素的Tensor变量，否则必须在  :math:`[−rank(input),rank(input)]` 范围内。如果 :math:`dim [i] <0` ，则维度将减小为 :math:`rank+dim[i]` 。
          - **keep_dim** （bool | False）：是否在输出Tensor中保留减小的维度。除非 ``keep_dim`` 为true，否则结果张量将比输入少一个维度。
          - **name** （str | None）：这一层的名称（可选）。如果设置为None，则将自动命名这一层。

返回：  运算、减少维度之后的Tensor变量。

返回类型：  变量（Variable）
          
**代码示例**

..  code-block:: python

      # x是一个Tensor，元素如下:
      #    [[0.2, 0.3, 0.5, 0.9]
      #     [0.1, 0.2, 0.6, 0.7]]
      # 接下来的示例中，我们在每处函数调用后面都标注出了它的结果张量。。
      fluid.layers.reduce_max(x)  # [0.9]
      fluid.layers.reduce_max(x, dim=0)  # [0.2, 0.3, 0.6, 0.9]
      fluid.layers.reduce_max(x, dim=-1)  # [0.9, 0.7]
      fluid.layers.reduce_max(x, dim=1, keep_dim=True)  # [[0.9], [0.7]]

      # x是一个shape为[2, 2, 2]的Tensor，元素如下:
      #      [[[1.0, 2.0], [3.0, 4.0]],
      #      [[5.0, 6.0], [7.0, 8.0]]]
      # 接下来的示例中，我们在每处函数调用后面都标注出了它的结果张量。。
      fluid.layers.reduce_max(x, dim=[1, 2]) # [4.0, 8.0]
      fluid.layers.reduce_max(x, dim=[0, 1]) # [7.0, 8.0]










.. _cn_api_fluid_layers_reduce_mean:

reduce_mean
-------------------------------

.. py:function:: paddle.fluid.layers.reduce_mean(input, dim=None, keep_dim=False, name=None)

计算给定维度上张量（Tensor）元素平均值。

参数：
          - **input** （Variable）：输入变量为Tensor或LoDTensor。
          - **dim** （list | int | None）：函数运算的维度。如果为None，则对输入的所有元素求平均值并返回单个元素的Tensor变量，否则必须在  :math:`[−rank(input),rank(input)]` 范围内。如果 :math:`dim [i] <0` ，则维度将减小为 :math:`rank+dim[i]` 。
          - **keep_dim** （bool | False）：是否在输出Tensor中保留减小的维度。除非 ``keep_dim`` 为true，否则结果张量将比输入少一个维度。
          - **name** （str | None）：这一层的名称（可选）。如果设置为None，则将自动命名这一层。

返回：  运算、减少维度之后的Tensor变量。

返回类型：  变量（Variable）
          
**代码示例**

..  code-block:: python

      # x是一个Tensor，元素如下:
      #    [[0.2, 0.3, 0.5, 0.9]
      #     [0.1, 0.2, 0.6, 0.7]]
      # 接下来的示例中，我们在每处函数调用后面都标注出了它的结果张量。
      fluid.layers.reduce_mean(x)  # [0.4375]
      fluid.layers.reduce_mean(x, dim=0)  # [0.15, 0.25, 0.55, 0.8]
      fluid.layers.reduce_mean(x, dim=-1)  # [0.475, 0.4]
      fluid.layers.reduce_mean(
          x, dim=1, keep_dim=True)  # [[0.475], [0.4]]

      # x 是一个shape为[2, 2, 2]的Tensor元素如下:
      #      [[[1.0, 2.0], [3.0, 4.0]],
      #      [[5.0, 6.0], [7.0, 8.0]]]
      # 接下来的示例中，我们在每处函数调用后面都标注出了它的结果张量。。
      fluid.layers.reduce_mean(x, dim=[1, 2]) # [2.5, 6.5]
      fluid.layers.reduce_mean(x, dim=[0, 1]) # [4.0, 5.0]










.. _cn_api_fluid_layers_reduce_min:

reduce_min
-------------------------------

.. py:function:: paddle.fluid.layers.reduce_min(input, dim=None, keep_dim=False, name=None)

计算给定维度上张量元素的最小值。

参数：
          - **input** （Variable）：输入变量为Tensor或LoDTensor。
          - **dim** （list | int | None）：函数运算的维度。如果为None，则对输入的所有元素做差并返回单个元素的Tensor变量，否则必须在  :math:`[−rank(input),rank(input)]` 范围内。如果 :math:`dim [i] <0` ，则维度将减小为 :math:`rank+dim[i]` 。
          - **keep_dim** （bool | False）：是否在输出Tensor中保留减小的维度。除非 ``keep_dim`` 为true，否则结果张量将比输入少一个维度。
          - **name** （str | None）：这一层的名称（可选）。如果设置为None，则将自动命名这一层。

返回：  运算、减少维度之后的Tensor变量。

返回类型：  变量（Variable）
          
**代码示例**

..  code-block:: python

      # x是一个Tensor，元素如下:
      #    [[0.2, 0.3, 0.5, 0.9]
      #     [0.1, 0.2, 0.6, 0.7]]
      # 接下来的示例中，我们在每处函数调用后面都标注出了它的结果张量。
      fluid.layers.reduce_min(x)  # [0.1]
      fluid.layers.reduce_min(x, dim=0)  # [0.1, 0.2, 0.5, 0.7]
      fluid.layers.reduce_min(x, dim=-1)  # [0.2, 0.1]
      fluid.layers.reduce_min(x, dim=1, keep_dim=True)  # [[0.2], [0.1]]

      # x 是一个shape为[2, 2, 2]的Tensor元素如下:
      #      [[[1.0, 2.0], [3.0, 4.0]],
      #      [[5.0, 6.0], [7.0, 8.0]]]
      # 接下来的示例中，我们在每处函数调用后面都标注出了它的结果张量。。
      fluid.layers.reduce_min(x, dim=[1, 2]) # [1.0, 5.0]
      fluid.layers.reduce_min(x, dim=[0, 1]) # [1.0, 2.0]










.. _cn_api_fluid_layers_reduce_prod:

reduce_prod
-------------------------------

.. py:function:: paddle.fluid.layers.reduce_prod(input, dim=None, keep_dim=False, name=None)

计算给定维度上张量（Tensor）元素乘积。

参数：
          - **input** （Variable）：输入变量为Tensor或LoDTensor。
          - **dim** （list | int | None）：函数运算的维度。如果为None，则将输入的所有元素相乘并返回单个元素的Tensor变量，否则必须在  :math:`[−rank(input),rank(input)]` 范围内。如果 :math:`dim [i] <0` ，则维度将减小为 :math:`rank+dim[i]` 。
          - **keep_dim** （bool | False）：是否在输出Tensor中保留减小的维度。除非 ``keep_dim`` 为true，否则结果张量将比输入少一个维度。
          - **name** （str | None）：这一层的名称（可选）。如果设置为None，则将自动命名这一层。

返回：  运算、减少维度之后的Tensor变量。

返回类型：  变量（Variable）
          
**代码示例**

..  code-block:: python

      # x是一个Tensor，元素如下:
      #    [[0.2, 0.3, 0.5, 0.9]
      #     [0.1, 0.2, 0.6, 0.7]]
      # 接下来的示例中，我们在每处函数调用后面都标注出了它的结果张量。
      fluid.layers.reduce_prod(x)  # [0.0002268]
      fluid.layers.reduce_prod(x, dim=0)  # [0.02, 0.06, 0.3, 0.63]
      fluid.layers.reduce_prod(x, dim=-1)  # [0.027, 0.0084]
      fluid.layers.reduce_prod(x, dim=1,
                               keep_dim=True)  # [[0.027], [0.0084]]

      # x 是一个shape为[2, 2, 2]的Tensor元素如下:
      #      [[[1.0, 2.0], [3.0, 4.0]],
      #      [[5.0, 6.0], [7.0, 8.0]]]
      # 接下来的示例中，我们在每处函数调用后面都标注出了它的结果张量。
      fluid.layers.reduce_prod(x, dim=[1, 2]) # [24.0, 1680.0]
      fluid.layers.reduce_prod(x, dim=[0, 1]) # [105.0, 384.0]










.. _cn_api_fluid_layers_reduce_sum:

reduce_sum
-------------------------------

.. py:function:: paddle.fluid.layers.reduce_sum(input, dim=None, keep_dim=False, name=None)

计算给定维度上张量（Tensor）元素之和。

参数：
          - **input** （Variable）- 输入变量为Tensor或LoDTensor。
          - **dim** （list | int | None）- 求和运算的维度。如果为None，则对输入的所有元素求和并返回单个元素的Tensor变量，否则必须在  :math:`[−rank(input),rank(input)]` 范围内。如果 :math:`dim [i] <0` ，则维度将减小为 :math:`rank+dim[i]` 。
          - **keep_dim** （bool | False）- 是否在输出Tensor中保留减小的维度。除非 ``keep_dim`` 为true，否则结果张量将比输入少一个维度。
          - **name** （str | None）- 这一层的名称（可选）。如果设置为None，则将自动命名这一层。

返回：  运算、减少维度之后的Tensor变量。

返回类型：  变量（Variable）
          
**代码示例**

..  code-block:: python

      # x是一个Tensor，元素如下:
      #    [[0.2, 0.3, 0.5, 0.9]
      #     [0.1, 0.2, 0.6, 0.7]]
      # 接下来的示例中，我们在每处函数调用后面都标注出了它的结果张量。
      fluid.layers.reduce_sum(x)  # [3.5]
      fluid.layers.reduce_sum(x, dim=0)  # [0.3, 0.5, 1.1, 1.6]
      fluid.layers.reduce_sum(x, dim=-1)  # [1.9, 1.6]
      fluid.layers.reduce_sum(x, dim=1, keep_dim=True)  # [[1.9], [1.6]]

      # x 是一个shape为[2, 2, 2]的Tensor元素如下:
      #      [[[1, 2], [3, 4]],
      #      [[5, 6], [7, 8]]]
      # 接下来的示例中，我们在每处函数调用后面都标注出了它的结果张量。
      fluid.layers.reduce_sum(x, dim=[1, 2]) # [10, 26]
      fluid.layers.reduce_sum(x, dim=[0, 1]) # [16, 20]
      









.. _cn_api_fluid_layers_relu:

relu
-------------------------------

.. py:function:: paddle.fluid.layers.relu(x, name=None)

Relu接受一个输入数据(张量)，输出一个张量。将线性函数y = max(0, x)应用到张量中的每个元素上。
    
.. math::                 
              \\Out=\max(0,x)\\
 

参数:
  - **x** (Variable):输入张量。
  - **name** (str|None，默认None) :如果设置为None，该层将自动命名。

返回: 与输入形状相同的输出张量。

返回类型: 变量（Variable）

**代码示例**:

..  code-block:: python

    output = fluid.layers.relu(x)










.. _cn_api_fluid_layers_relu6:

relu6
-------------------------------

.. py:function:: paddle.fluid.layers.relu6(x, threshold=6.0, name=None)

relu6激活算子（Relu6 Activation Operator）

.. math::
  
    \\out=min(max(0, x), 6)\\


参数:
    - **x** (Variable) - Relu6 operator的输入
    - **threshold** (FLOAT|6.0) - Relu6的阈值
    - **name** (str|None) -这个层的名称(可选)。如果设置为None，该层将被自动命名。

返回: Relu6操作符的输出

返回类型: 输出(Variable)


**代码示例：**

.. code-block:: python

    x = fluid.layers.data(name="x", shape=[3,10,32,32], dtype="float32")
    y = fluid.layers.relu6(x, threshold=6.0)







.. _cn_api_fluid_layers_reshape:

reshape
-------------------------------

.. py:function::  paddle.fluid.layers.reshape(x, shape, actual_shape=None, act=None, inplace=False, name=None)

保持输入张量数据不变的情况下，改变张量的形状。

目标形状可由 ``shape`` 或 ``actual_shape`` 给出。``shape`` 是一个整数列表，而 ``actual_shape`` 是一个张量变量。
当两个属性同时被指定时，``actual_shape`` 的优先级高于 ``shape`` ，但在编译时仍然应该正确地设置 ``shape`` 以保证形状推断。

在指定目标shape时存在一些技巧：

.. code-block:: text
	
	1. -1表示这个维度的值是从x的元素总数和剩余维度推断出来的。因此，有且只有一个维度可以被设置为-1。
	2. 0表示实际的维数是从x的对应维数中复制出来的，因此shape中0的索引值不能超过秩(x)。


这里有一些例子来解释它们：

.. code-block:: text

	1. 给定一个形状为[2,4,6]的三维张量x，目标形状为[6,8]， ``reshape`` 将x变换为形状为[6,8]的二维张量，且x的数据保持不变。
	2. 给定一个形状为[2,4,6]的三维张量x，指定的目标形状为[2,3,-1,2]， ``reshape``将x变换为形状为[2,3,4,2]的4- d张量，不改变x的数据。在这种情况下，目标形状的一个维度被设置为-1，这个维度的值是从x的元素总数和剩余维度推断出来的。
	3. 给定一个形状为[2,4,6]的三维张量x，目标形状为[- 1,0,3,2]，整形算子将x变换为形状为[2,4,3,2]的四维张量，使x的数据保持不变。在这种情况下，0意味着实际的维值将从x的对应维数中复制,-1位置的维度由x的元素总数和剩余维度计算得来。

参数：
	- **x** (variable) - 输入张量
	- **shape** (list) - 新的形状。新形状最多只能有一个维度为-1。
	- **actual_shape** (variable) - 一个可选的输入。如果提供，则根据 ``actual_shape`` 进行 reshape，而不是指定 ``shape`` 。也就是说，actual_shape具有比shape更高的优先级。
	- **act** (str) - 对reshpe后的tensor变量执行非线性激活
	- **inplace** (bool) - 如果在多个操作符中使用x，则 ``inplace`` 必须设置为False。如果该标志设置为True，则重用输入x进行reshape，这将改变张量变量x的形状，并可能在多个操作符中使用x时造成错误。如果为False，则保留形状x，并创建一个新的输出张量变量，该张量变量的数据是从输入x复制的，但经过了重构。
	- **name** (str) -  可选变量，此层的名称

返回：如果 ``act`` 为 ``None``,返回reshape后的tensor变量。如果 ``inplace`` 为 ``False`` ,将返回一个新的Tensor变量，否则，将改变x自身。如果 ``act`` 不是 ``None`` ，则返回激活的张量变量。

抛出异常：``TypeError`` - 如果 actual_shape 既不是变量也不是None

**代码示例**

.. code-block:: python

	data = fluid.layers.data(
	    name='data', shape=[2, 4, 6], dtype='float32')
	reshaped = fluid.layers.reshape(
	    x=data, shape=[-1, 0, 3, 2], inplace=True)










.. _cn_api_fluid_layers_resize_bilinear:

resize_bilinear
-------------------------------

.. py:function:: paddle.fluid.layers.resize_bilinear(input, out_shape=None, scale=None, name=None)

根据指定的out_shape执行双线性插值调整输入大小，输出形状按优先级由actual_shape、out_shape和scale指定。

双线性插值是对线性插值的扩展,即二维变量方向上(如h方向和w方向)插值。关键思想是先在一个方向上执行线性插值，然后再在另一个方向上执行线性插值。

 `详情请参阅维基百科 https://en.wikipedia.org/wiki/Bilinear_interpolation <https://en.wikipedia.org/wiki/Bilinear_interpolation>`_ 

参数:
        - **input** (Variable) - 双线性插值的输入张量，是一个shpae为(N x C x h x w)的4d张量。
        - **out_shape** (Variable) - 一维张量，包含两个数。第一个数是高度，第二个数是宽度。
        - **scale** (float|None) - 用于输入高度或宽度的乘数因子。out_shape和scale至少要设置一个。out_shape的优先级高于scale。默认值:None。
        - **name** (str|None) - 输出变量名。
        - **actual_shape** (Variable) - 可选输入，用于动态指定输出形状。如果指定actual_shape，图像将根据给定的形状调整大小，而不是根据指定形状的 :code:`out_shape` 和 :code:`scale` 进行调整。也就是说， :code:`actual_shape` 具有最高的优先级。如果希望动态指定输出形状，建议使用 :code:`actual_shape` 而不是 :code:`out_shape` 。在使用actual_shape指定输出形状时，还需要设置out_shape和scale之一，否则在图形构建阶段会出现错误。默认值:None

返回：	输出的维度是(N x C x out_h x out_w)


**代码示例：**

.. code-block:: python

  out = fluid.layers.resize_bilinear(input, out_shape=[12, 12])








.. _cn_api_fluid_layers_resize_nearest:

resize_nearest
-------------------------------

.. py:function:: paddle.fluid.layers.resize_nearest(input, out_shape=None, scale=None, name=None, actual_shape=None)

该层对输入进行放缩，在第三维（高度方向）和第四维（宽度方向）进行最邻近插值（nearest neighbor interpolation）操作。
输出形状按优先级顺序依据 ``actual_shape`` , ``out_shape`` 和 ``scale`` 而定。

最邻近插值的详细介绍请参照： `Wiki Nearest-neighbor interpolation <https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation>`_



参数:
  - **input** (Variable) – 插值运算的输入张量, 是一个形为 (N,C,H,W) 的四维张量
  - **out_shape** (Variable) – 一维张量，包含两个指明输出大小的数字 。 第一个代表了高度，第二个代表了宽度
  - **scale** (float|None) – 输入高、宽的乘法器。 ``out_shape`` 和 ``scale`` 二者至少设置其一。 ``out_shape`` 具有比 ``scale`` 更高的优先级。 默认: None
  - **name** (str|None) – 输出变量的命名
  - **actual_shape** (Variable) – 可选输入， 动态设置输出张量的形状。 如果提供该值， 图片放缩会依据此形状进行， 而非依据 ``out_shape`` 和 ``scale`` 。 即为， ``actual_shape`` 具有最高的优先级。 如果想动态指明输出形状，推荐使用 ``actual_shape`` 取代 ``out_shape`` 。 当使用 ``actual_shape`` 来指明输出形状， ``out_shape`` 和 ``scale`` 也应该进行设置, 否则在图形生成阶段将会报错。默认: None

返回：插值运算的输出张量，是一个形为 (N,C,H,W) 的四维张量

**代码示例**

..  code-block:: python

    out = fluid.layers.resize_nearest(input, out_shape=[12, 12])










.. _cn_api_fluid_layers_roi_align:

roi_align
-------------------------------

.. py:function:: paddle.fluid.layers.roi_align(input, rois, pooled_height=1, pooled_width=1, spatial_scale=1.0, sampling_ratio=-1, name=None)

**实现RoIAlign操作。**

Region of Interests align(直译：有意义、有价值选区对齐) 用于实现双线性插值，它可以将不均匀大小的输入
变为固定大小的特征图(feature map)。

该运算通过 ``pooled_width`` 和 ``pooled_height`` 将每个推荐区域划分为等大小分块。位置保持不变。

在每个RoI框中，四个常取样位置会通过双线性插值直接计算。输出为这四个位置的平均值从而解决不对齐问题。

参数:
  - **input** (Variable) – (Tensor) 该运算的的输入张量，形为(N,C,H,W)。其中 N 为batch大小, C 为输入通道的个数, H 特征高度, W 特征宽度
  - **rois** (Variable) – 待池化的ROIs (Regions of Interest)
  - **pooled_height** (integer) – (默认为1), 池化后的输出高度
  - **pooled_width** (integer) – (默认为1), 池化后的输出宽度
  - **spatial_scale** (float) – (默认为1.0),乘法性质空间标尺因子，池化时，将RoI坐标变换至运算采用的标度
  - **sampling_ratio** (intger) – (默认为-1),插值格中采样点的数目。 如果它 <=0, 它们将自适应 ``roi_width`` 和 ``pooled_w`` , 在高度上也是同样的道理。

返回：一个形为 (num_rois, channels, pooled_h, pooled_w) 的四维张量

返回类型：Variable

**代码示例**

..  code-block:: python

    align_out = fluid.layers.roi_align(input=x,
                                       rois=rois,
                                       pooled_height=7,
                                       pooled_width=7,
                                       spatial_scale=0.5,
                                       sampling_ratio=-1)











.. _cn_api_fluid_layers_roi_pool:

roi_pool
-------------------------------

.. py:function:: paddle.fluid.layers.roi_pool(input, rois, pooled_height=1, pooled_width=1, spatial_scale=1.0)

    
roi池化是对非均匀大小的输入执行最大池化，以获得固定大小的特征映射(例如7*7)。
    
该operator有三个步骤:

    1. 用pooled_width和pooled_height将每个区域划分为大小相等的部分
    2. 在每个部分中找到最大的值
    3. 将这些最大值复制到输出缓冲区

Faster-RCNN.使用了roi池化。roi关于roi池化请参考 https://stackoverflow.com/questions/43430056/what-is-roi-layer-in-fast-rcnn

参数:    
    - **input** (Variable) - 张量，ROIPoolOp的输入。输入张量的格式是NCHW。其中N为batch大小，C为输入通道数，H为特征高度，W为特征宽度
    - **roi** (Variable) -  roi区域。
    - **pooled_height** (integer) - (int，默认1)，池化输出的高度。默认:1
    - **pooled_width** (integer) -  (int，默认1) 池化输出的宽度。默认:1
    - **spatial_scale** (float) - (float，默认1.0)，用于将ROI coords从输入规模转换为池化时使用的规模。默认1.0

返回: (张量)，ROIPoolOp的输出是一个shape为(num_rois, channel, pooled_h, pooled_w)的4d张量。
    
返回类型: 变量（Variable）
    

**代码示例**

..  code-block:: python

	pool_out = fluid.layers。roi_pool(输入=x, rois=rois, 7,7,1.0)










.. _cn_api_fluid_layers_row_conv:

row_conv
-------------------------------

.. py:function:: paddle.fluid.layers.row_conv(input, future_context_size, param_attr=None, act=None)

行卷积（Row-convolution operator）称为超前卷积（lookahead convolution）。下面关于DeepSpeech2的paper中介绍了这个operator 
    
    `<http://www.cs.cmu.edu/~dyogatam/papers/wang+etal.iclrworkshop2016.pdf>`_ 

双向的RNN在深度语音模型中很有用，它通过对整个序列执行正向和反向传递来学习序列的表示。然而，与单向RNNs不同的是，在线部署和低延迟设置中，双向RNNs具有难度。超前卷积将来自未来子序列的信息以一种高效的方式进行计算，以改进单向递归神经网络。 row convolution operator 与一维序列卷积不同，计算方法如下:
   
给定输入序列长度为 :math:`t` 的输入序列 :math:`in` 和输入维度 :math:`d` ，以及一个大小为 :math:`context x d` 的滤波器 :math:`W` ，输出序列卷积为:

.. math::   
		out_i = \sum_{j=i}^{i+context} in_{j} · W_{i-j}
    
公式中：
    - :math:`out_i` : 第i行输出变量 shaoe为[1, D].
    - :math:`context` ： 未来上下文（feature context）大小
    - :math:`in_j` : 第j行输出变量,形为[1，D]
    - :math:`W_{i-j}` : 第(i-j)行参数，其形状为[1,D]。

`详细请参考设计文档 https://github.com/PaddlePaddle/Paddle/issues/2228#issuecomment-303903645 <https://github.com/PaddlePaddle/Paddle/issues/2228#issuecomment-303903645>`_  .

参数:
    - **input** (Variable) -- 输入是一个LodTensor，它支持可变时间长度的输入序列。这个LodTensor的内部张量是一个具有形状(T x N)的矩阵，其中T是这个mini batch中的总的timestep，N是输入数据维数。
    - **future_context_size** (int) -- 未来上下文大小。请注意，卷积核的shape是[future_context_size + 1, D]。
    - **param_attr** (ParamAttr) --  参数的属性，包括名称、初始化器等。
    - **act** (str) -- 非线性激活函数。
    
返回: 输出(Out)是一个LodTensor，它支持可变时间长度的输入序列。这个LodTensor的内部量是一个形状为 T x N 的矩阵，和X的 shape 一样。


**代码示例**

..  code-block:: python

	import paddle.fluid as fluid
     
     	x = fluid.layers.data(name='x', shape=[16],
                        dtype='float32', lod_level=1)
	out = fluid.layers.row_conv(input=x, future_context_size=2)










.. _cn_api_fluid_layers_sampling_id:

sampling_id
-------------------------------

.. py:function:: paddle.fluid.layers.sampling_id(x, min=0.0, max=1.0, seed=0, dtype='float32')

sampling_id算子。用于从输入的多项分布中对id进行采样的图层。为一个样本采样一个id。

参数：
        - **x** （Variable）- softmax的输入张量（Tensor）。2-D形状[batch_size，input_feature_dimensions]
        - **min** （Float）- 随机的最小值。（浮点数，默认为0.0）
        - **max** （Float）- 随机的最大值。（float，默认1.0）
        - **seed** （Float）- 用于随机数引擎的随机种子。0表示使用系统生成的种子。请注意，如果seed不为0，则此运算符将始终每次生成相同的随机数。（int，默认为0）
        - **dtype** （np.dtype | core.VarDesc.VarType | str）- 输出数据的类型为float32，float_16，int等。

返回：       Id采样的数据张量。

返回类型：        输出（Variable）。


 








.. _cn_api_fluid_layers_scale:

scale
-------------------------------

.. py:function:: paddle.fluid.layers.scale(x, scale=1.0, bias=0.0, bias_after_scale=True, act=None, name=None)

缩放算子

对输入张量应用缩放和偏移加法。

if ``bias_after_scale`` = True:

.. math::
                                Out=scale*X+bias

else:

.. math::
                                Out=scale*(X+bias)

参数:
        - **x** (Variable) - (Tensor) 要比例运算的输入张量（Tensor）。
        - **scale** (FLOAT) - 比例运算的比例因子。
        - **bias** (FLOAT) - 比例算子的偏差。
        - **bias_after_scale** (BOOLEAN) - 在缩放之后或之前添加bias。在某些情况下，对数值稳定性很有用。
        - **act** (basestring|None) - 应用于输出的激活函数。
        - **name** (basestring|None)- 输出的名称。

返回:        比例运算符的输出张量(Tensor)

返回类型:        变量(Variable)










.. _cn_api_fluid_layers_scatter:

scatter
-------------------------------

.. py:function:: paddle.fluid.layers.scatter(input, index, updates, name=None)


通过更新输入在第一维度上指定索引位置处的元素来获得输出。

.. math::
          \\Out=XOut[Ids]=Updates\\


参数：
  - **input** （Variable） - 秩> = 1的源输入
  - **index** （Variable） - 秩= 1的索引输入。 它的dtype应该是int32或int64，因为它用作索引
  - **updates** （Variable） - scatter 要进行更新的变量
  - **name** （str | None） - 输出变量名称。 默认None

返回：张量变量, 与输入张量的shape相同

返回类型：output（Variable）

**代码示例**

..  code-block:: python

    output = fluid.layers.scatter(input, index, updates)











.. _cn_api_fluid_layers_selu:

selu
-------------------------------

.. py:function:: paddle.fluid.layers.selu(x, scale=None, alpha=None, name=None)

**实现Selu运算**

有如下等式：

.. math::
    selu= \lambda*
    \begin{cases}
         x                      &\quad \text{ if } x>0 \\ 
         \alpha * e^x - \alpha  &\quad \text{ if } x<=0 
    \end{cases}

输入 ``x`` 可以选择性携带LoD信息。输出和它共享此LoD信息(如果有)。

参数:
  - **x** (Variable) – 输入张量
  - **scale** (float, None) – 如果标度没有设置，其默认值为 1.0507009873554804934193349852946。 详情请见： `Self-Normalizing Neural Networks <https://arxiv.org/abs/1706.02515.pdf>`_ 
  - **alpha** (float, None) – 如果没有设置改参数, 其默认值为 1.6732632423543772848170429916717。 详情请见： `Self-Normalizing Neural Networks <https://arxiv.org/abs/1706.02515.pdf>`_
  - **name** (str|None, default None) – 该层命名，若为None则自动为其命名

返回：一个形和输入张量相同的输出张量

返回类型：Variable

**代码示例**

..  code-block:: python

    output = fluid.layers.selu(x)













.. _cn_api_fluid_layers_sequence_concat:

sequence_concat
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_concat(input, name=None)

sequence_concat操作通过序列信息连接LoD张量（Tensor）。例如：X1的LoD = [0,3,7]，X2的LoD = [0,7,9]，结果的LoD为[0，（3 + 7），（7 + 9）]，即[0,10,16]。

参数:
        - **input** (list) – List of Variables to be concatenated.
        - **name** (str|None) – A name for this layer(optional). If set None, the layer will be named automatically.
        
返回:     连接好的输出变量。

返回类型:   变量（Variable）


**代码示例**

..  code-block:: python

        out = fluid.layers.sequence_concat(input=[seq1, seq2, seq3])
        









.. _cn_api_fluid_layers_sequence_conv:

sequence_conv 
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_conv(input, num_filters, filter_size=3, filter_stride=1, padding=None, bias_attr=None, param_attr=None, act=None, name=None)

该函数的输入参数中给出了滤波器和步长，通过利用输入以及滤波器和步长的常规配置来为sequence_conv创建操作符。

参数：
    - **input** (Variable) - (LoD张量）输入X是LoD张量，支持可变的时间量的长度输入序列。该LoDTensor的标记张量是一个维度为（T,N)的矩阵，其中T是mini-batch的总时间步数，N是input_hidden_size
    - **num_filters** (int) - 滤波器的数量
    - **filter_size** (int) - 滤波器大小（H和W)
    - **filter_stride** (int) - 滤波器的步长
    - **padding** (bool) - 若为真，添加填充
    - **bias_attr** (ParamAttr|bool|None) - sequence_conv偏离率参数属性。若设为False,输出单元则不加入偏离率。若设为None或ParamAttr的一个属性，sequence_conv将创建一个ParamAttr作为bias_attr。如果未设置bias_attr的初始化函数，则将bias初始化为0.默认:None
    - **param_attr** (ParamAttr|None) - 可学习参数/sequence_conv的权重参数属性。若设置为None或ParamAttr的一个属性，sequence_conv将创建ParamAttr作为param_attr。
    若未设置param_attr的初始化函数，则用Xavier初始化参数。默认:None

返回：sequence_conv的输出

返回类型：变量（Variable）









.. _cn_api_fluid_layers_sequence_enumerate:

sequence_enumerate
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_enumerate(input, win_size, pad_value=0, name=None)

为输入索引序列生成一个新序列，该序列枚举输入长度为 ``win_size`` 的所有子序列。 输入序列和枚举序列第一维上维度相同，第二维是 ``win_size`` ，在生成中如果需要，通过设置 ``pad_value`` 填充。

**例子：**

::

        输入：
            X.lod = [[0, 3, 5]]  X.data = [[1], [2], [3], [4], [5]]  X.dims = [5, 1]
        属性：
            win_size = 2  pad_value = 0
        输出：
            Out.lod = [[0, 3, 5]]  Out.data = [[1, 2], [2, 3], [3, 0], [4, 5], [5, 0]]  Out.dims = [5, 2]
        
参数:   
        - **input** （Variable）- 作为索引序列的输入变量。
        - **win_size** （int）- 枚举所有子序列的窗口大小。
        - **pad_value** （int）- 填充值，默认为0。
          
返回:      枚举序列变量是LoD张量（LoDTensor）。

返回类型:   Variable
          
**代码示例**

..  code-block:: python

      x = fluid.layers.data(shape[30, 1], dtype='int32', lod_level=1)
      out = fluid.layers.sequence_enumerate(input=x, win_size=3, pad_value=0)









.. _cn_api_fluid_layers_sequence_expand:

sequence_expand 
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_expand(x, y, ref_level=-1, name=None)

序列扩张层（Sequence Expand Layer)

将根据指定 y 的 level lod 展开输入变量x，请注意 x 的 lod level 最多为1，而 x 的秩最少为2。当 x 的秩大于2时，它就被看作是一个二维张量。下面的例子将解释 sequence_expand 是如何工作的:

::


    * 例1
	    x is a LoDTensor:
		x.lod  = [[2,        2]]
		x.data = [[a], [b], [c], [d]]
		x.dims = [4, 1]

	    y is a LoDTensor:
		y.lod = [[2,    2],
		         [3, 3, 1, 1]]

	    ref_level: 0

	    then output is a 1-level LoDTensor:
		out.lod =  [[2,        2,        2,        2]]
		out.data = [[a], [b], [a], [b], [c], [d], [c], [d]]
		out.dims = [8, 1]

    * 例2
	    x is a Tensor:
		x.data = [[a], [b], [c]]
		x.dims = [3, 1]

	    y is a LoDTensor:
		y.lod = [[2, 0, 3]]

	    ref_level: -1

	    then output is a Tensor:
		out.data = [[a], [a], [c], [c], [c]]
		out.dims = [5, 1]

参数：
    - **x** (Variable) - 输入变量，张量或LoDTensor
    - **y** (Variable) - 输入变量，为LoDTensor
    - **ref_level** (int) - x表示的y的Lod层。若设为-1，表示lod的最后一层
    - **name** (str|None) - 该层名称（可选）。如果设为空，则自动为该层命名

返回：扩展变量，LoDTensor

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    x = fluid.layers.data(name='x', shape=[10], dtype='float32')
    y = fluid.layers.data(name='y', shape=[10, 20],
                 dtype='float32', lod_level=1)
    out = layers.sequence_expand(x=x, y=y, ref_level=0)









.. _cn_api_fluid_layers_sequence_expand_as:

sequence_expand_as 
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_expand_as(x, y, name=None)

Sequence Expand As Layer

这一层将根据y的第0级lod展开输入变量x。当前实现要求输入（Y）的lod层数必须为1，输入（X）的第一维应当和输入（Y）的第0层lod的大小相同，不考虑输入（X）的lod。

以下示例解释sequence_expand如何工作：

::

    * 例1:
    给定一维LoDTensor input(X)
        X.data = [[a], [b], [c], [d]]
        X.dims = [4, 1]
    和 input(Y)
        Y.lod = [[0, 3, 6, 7, 8]]
    ref_level: 0
    得到1级 LoDTensor
        Out.lod =  [[0,            3,              6,  7,  8]]
        Out.data = [[a], [a], [a], [b], [b], [b], [c], [d]]
        Out.dims = [8, 1]

参数：
    - **x** (Variable) - 输入变量，类型为Tensor或LoDTensor
    - **y** (Variable) - 输入变量，为LoDTensor
    - **name** (str|None) - 该层名称（可选）。如果设为空，则自动为该层命名

返回：扩展变量，LoDTensor

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    x = fluid.layers.data(name='x', shape=[10], dtype='float32')
    y = fluid.layers.data(name='y', shape=[10, 20],
                 dtype='float32', lod_level=1)
    out = layers.sequence_expand_as(x=x, y=y)









.. _cn_api_fluid_layers_sequence_first_step:

sequence_first_step
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_first_step(input)

该功能获取序列的第一步

::

    x是1-level LoDTensor:

      x.lod = [[2, 3, 2]]

      x.data = [1, 3, 2, 4, 6, 5, 1]

      x.dims = [7, 1]

    输出为张量:

      out.dim = [3, 1]
      with condition len(x.lod[-1]) == out.dims[0]
      out.data = [1, 2, 5], where 1=first(1,3), 2=first(2,4,6), 5=first(5,1)

参数：**input** (variable)-输入变量，为LoDTensor

返回：序列第一步，为张量

**代码示例**：

.. code-block:: python

    x = fluid.layers.data(name='x', shape=[7, 1],
                 dtype='float32', lod_level=1)
    x_first_step = fluid.layers.sequence_first_step(input=x)









.. _cn_api_fluid_layers_sequence_last_step:

sequence_last_step
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_last_step(input)

该API可以获取序列的最后一步

::

    x是level-1的LoDTensor:

        x.lod = [[2, 3, 2]]

        x.data = [1, 3, 2, 4, 6, 5, 1]

        x.dims = [7, 1]

    输出为Tensor:

        out.dim = [3, 1]
        
        且 len(x.lod[-1]) == out.dims[0]
        
        out.data = [1, 2, 5], where 1=first(1,3), 2=first(2,4,6), 5=first(5,1)

参数：**input** (variable)-输入变量，为LoDTensor

返回：序列的最后一步，为张量

**代码示例**：

.. code-block:: python

    x = fluid.layers.data(name='x', shape=[7, 1],
                 dtype='float32', lod_level=1)
    x_last_step = fluid.layers.sequence_last_step(input=x)









.. _cn_api_fluid_layers_sequence_mask:

sequence_mask
-------------------------------

该层根据输入 ``x`` 和 ```maxlen`` 输出一个掩码，数据类型为dtype。

假设x是一个形状为[d_1, d_2，…]的张量。， y是一个形为[d_1, d_2，… ，d_n, maxlen]的掩码，其中:

.. math::

	y(i_1, i_2,..., i_n, j) = (j < x(i_1, i_2,..., i_n))

参数：
	- **x** (Variable) - sequence_mask层的输入张量，其元素是小于maxlen的整数。
	- **maxlen** (int|None) - 序列的最大长度。如果maxlen为空，则用max(x)替换。
	- **dtype** (np.dtype|core.VarDesc.VarType|str) - 输出的数据类型
	- **name** (str|None) - 此层的名称(可选)。如果没有设置，该层将被自动命名。

返回： sequence mask 的输出

返回类型： Variable












.. _cn_api_fluid_layers_sequence_pad:

sequence_pad
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_pad(x,pad_value,maxlen=None,name=None)

序列填充操作符（Sequence Pad Operator）

这个操作符将同一batch中的序列填充到一个一致的长度。长度由属性padded_length指定。填充的新元素的值具体由输入 ``PadValue`` 指定，并会添加到每一个序列的末尾，使得他们最终的长度保持一致。

以下的例子更清晰地解释此操作符的工作原理：

::

    例1:

    给定 1-level LoDTensor
    
    input(X):
        X.lod = [[0,2,5]]
        X.data = [a,b,c,d,e]
    input(PadValue):
        PadValue.data = [0]
    
    'padded_length'=4

    得到LoDTensor:
        Out.data = [[a,b,0,0],[c,d,e,0]]
        Length.data = [[2],[3]]

::

    例2:
    
    给定 1-level LoDTensor
    
    input(X):
        X.lod = [[0,2,5]]
        X.data = [[a1,a2],[b1,b2],[c1,c2],[d1,d2],[e1,e2]]
    input(PadValue):
        PadValue.data = [0]
    
    'padded_length' = -1,表示用最长输入序列的长度(此例中为3)
    
    得到LoDTensor:
        Out.data = [[[a1,a2],[b1,b2],[0,0]],[[c1,c2],[d1,d2],[e1,e2]]]
        Length.data = [[2],[3]]


::

    例3:
    
    给定 1-level LoDTensor
    
    input(X):
        X.lod = [[0,2,5]]
        X.data = [[a1,a2],[b1,b2],[c1,c2],[d1,d2],[e1,e2]]
    input(PadValue):
        PadValue.data = [p1,p2]
    
    'padded_length' = -1,表示用最长输入序列的长度（此例中为3）
    
    得到LoDTensor:
        Out.data = [[[a1,a2],[b1,b2],[p1,p2]],[[c1,c2],[d1,d2],[e1,e2]]]
        Length.data = [[2],[3]]


参数：
    - **x** (Vairable) - 输入变量，应包含lod信息
    - **pad_value** (Variable) - 变量，存有放入填充步的值。可以是标量或tensor,维度和序列的时间步长相等。如果是标量,则自动广播到时间步长的维度
    - **maxlen** (int,默认None) - 填充序列的长度。可以为空或者任意正整数。当为空时，以序列中最长序列的长度为准，其他所有序列填充至该长度。当是某个特定的正整数，最大长度必须大于最长初始序列的长度
    - **name** (str|None) – 该层的命名(可选项)。 如果为 None, 则自动命名

返回：填充序列批（batch）和填充前的初始长度。所有输出序列的长度相等

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import numpy

    x = fluid.layers.data(name='y', shape=[10, 5],
                     dtype='float32', lod_level=1)
    pad_value = fluid.layers.assign(
        input=numpy.array([0.0], dtype=numpy.float32))
    out = fluid.layers.sequence_pad(x=x, pad_value=pad_value)









.. _cn_api_fluid_layers_sequence_pool:

sequence_pool 
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_pool(input, pool_type, is_test=False)

该函数为序列的池化添加操作符。将每个实例的所有时间步数特征池化，并用参数中提到的pool_type将特征运用到输入到首部。

支持四种pool_type:

- **average**: :math:`Out[i] = \frac{\sum_{i}X_{i}}{N}`
- **sum**: :math:`Out[i] = \sum _{j}X_{ij}`
- **sqrt**: :math:`Out[i] = \frac{ \sum _{j}X_{ij}}{\sqrt{len(\sqrt{X_{i}})}}`
- **max**: :math:`Out[i] = max(X_{i})`

::


    x是一级LoDTensor:
        x.lod = [[2, 3, 2]]
        x.data = [1, 3, 2, 4, 6, 5, 1]
        x.dims = [7, 1]
    输出为张量（Tensor）：
        out.dim = [3, 1]
        with condition len(x.lod[-1]) == out.dims[0]
    对于不同的pool_type：
        average: out.data = [2, 4, 3], where 2=(1+3)/2, 4=(2+4+6)/3, 3=(5+1)/2
        sum    : out.data = [4, 12, 6], where 4=1+3, 12=2+4+6, 6=5+1
        sqrt   : out.data = [2.82, 6.93, 4.24], where 2.82=(1+3)/sqrt(2),
             6.93=(2+4+6)/sqrt(3), 4.24=(5+1)/sqrt(2)
        max    : out.data = [3, 6, 5], where 3=max(1,3), 6=max(2,4,6), 5=max(5,1)
        last   : out.data = [3, 6, 1], where 3=last(1,3), 6=last(2,4,6), 1=last(5,1)
        first  : out.data = [1, 2, 5], where 1=first(1,3), 2=first(2,4,6), 5=first(5,1)

参数：
    - **input** (variable) - 输入变量，为LoDTensor
    - **pool_type** (string) - 池化类型。支持average,sum,sqrt和max
    - **is_test** (bool, 默认为 False) - 用于区分训练模式和测试评分模式

返回：sequence pooling 变量，类型为张量（Tensor)

**代码示例**:

.. code-block:: python

    x = fluid.layers.data(name='x', shape=[7, 1],
                 dtype='float32', lod_level=1)
    avg_x = fluid.layers.sequence_pool(input=x, pool_type='average')
    sum_x = fluid.layers.sequence_pool(input=x, pool_type='sum')
    sqrt_x = fluid.layers.sequence_pool(input=x, pool_type='sqrt')
    max_x = fluid.layers.sequence_pool(input=x, pool_type='max')
    last_x = fluid.layers.sequence_pool(input=x, pool_type='last')
    first_x = fluid.layers.sequence_pool(input=x, pool_type='first')









.. _cn_api_fluid_layers_sequence_reshape:

sequence_reshape
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_reshape(input, new_dim) 

Sequence Reshape Layer
该层重排输入序列。用户设置新维度。每一个序列的的长度通过原始长度、原始维度和新的维度计算得出。以下实例帮助解释该层的功能

.. code-block:: python

    x是一个LoDTensor:
        x.lod  = [[0, 2, 6]]
        x.data = [[1,  2], [3,  4],
                [5,  6], [7,  8],
                [9, 10], [11, 12]]
        x.dims = [6, 2]
    设置 new_dim = 4
    输出为LoDTensor:
        out.lod  = [[0, 1, 3]]

        out.data = [[1,  2,  3,  4],
                    [5,  6,  7,  8],
                    [9, 10, 11, 12]]
        out.dims = [3, 4]

目前仅提供1-level LoDTensor，请确保(原长度*原维数)可以除以新的维数，每个序列没有余数。

参数：
    - **input** (Variable)-一个2-D LoDTensor,模型为[N,M]，维度为M
    - **new_dim** (int)-新维度，输入LoDTensor重新塑造后的新维度

返回：根据新维度重新塑造的LoDTensor

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    x = fluid.layers.data(shape=[5, 20], dtype='float32', lod_level=1)
    x_reshaped = fluid.layers.sequence_reshape(input=x, new_dim=10)









.. _cn_api_fluid_layers_sequence_reverse:

sequence_reverse
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_reverse(x, name=None)

**实现SequenceReverse（序列倒序）运算**

在第0维上将输入 ``x`` 的各序列倒序。

::

    假设 ``x`` 是一个形为 (5,4) 的LoDTensor， lod信息为 [[0, 2, 5]]，其中，


    X.data() = [ [1, 2, 3, 4], [5, 6, 7, 8], # 索引为0，长度为2的序列

                 [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20] # 索引为1长度为3的序列

输出 ``Y`` 与 ``x`` 具有同样的维数和LoD信息。 于是有：

::

    Y.data() = [ [5, 6, 7, 8], [1, 2, 3, 4], # 索引为0，长度为2的逆序列 
                 [17, 18, 19, 20], [13, 14, 15, 16], [9, 10, 11, 12] # 索引为1，长度为3的逆序列

该运算在建立反dynamic RNN 网络中十分有用。

目前仅支持LoD层次(LoD level)为1的张量倒序。

参数:
  - **x** (Variable) – 输入张量
  - **name** (basestring|None) – 输出变量的命名

返回：输出LoD张量

返回类型：Variable











.. _cn_api_fluid_layers_sequence_scatter:

sequence_scatter
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_scatter(input, index, updates, name=None)

序列散射层

这个operator将更新张量X，它使用Ids的LoD信息来选择要更新的行，并使用Ids中的值作为列来更新X的每一行。

**样例**:
 
::

    输入：
    input.data = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    input.dims = [3, 6]

    index.data = [[0], [1], [2], [5], [4], [3], [2], [1], [3], [2], [5], [4]] index.lod = [[0, 3, 8, 12]]

    updates.data = [[0.3], [0.3], [0.4], [0.1], [0.2], [0.3], [0.4], [0.0], [0.2], [0.3], [0.1], [0.4]] updates.lod = [[ 0, 3, 8, 12]]


    输出：
    out.data = [[1.3, 1.3, 1.4, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.4, 1.3, 1.2, 1.1], [1.0, 1.0, 1.3, 1.2, 1.4, 1.1]]
    out.dims = X.dims = [3, 6]


参数：
      - **input** (Variable) - input 秩（rank） >= 1。
      - **index** (Variable) - index 秩（rank）=1。由于用于索引dtype应该是int32或int64。
      - **updates** (Variable) - input需要被更新的值。
      - **name** (str|None) - 输出变量名。默认：None。

返回： 输出张量维度应该和输入张量相同

返回类型：output (Variable)


**代码示例**:

..  code-block:: python

  output = fluid.layers.sequence_scatter(input, index, updates)










.. _cn_api_fluid_layers_sequence_slice:

sequence_slice
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_slice(input, offset, length, name=None)

**实现Sequence Slice(序列切片)运算**

该层从给定序列中截取子序列。截取依据为所给的开始 ``offset`` （偏移量） 和子序列长 ``length`` 。

仅支持序列数据，LoD level（LoD层次为1）
::
    输入变量：

        input.data = [[a1, a2], [b1, b2], [c1, c2], [d1, d2], [e1, e2]],
        input.lod = [[3, 2]],
        input.dims = (5, 2),

    以及 offset.data = [[0], [1]] and length.data = [[2], [1]],

    则输出变量为：

        out.data = [[a1, a2], [b1, b2], [e1, e2]],
        out.lod = [[2, 1]],
        out.dims = (3, 2).

.. note::
   ``input`` ， ``offset`` ， ``length`` 的第一维大小应相同。 
   ``offset`` 从0开始。

参数:
  - **input** (Variable) – 输入变量 ，承载着完整的序列
  - **offset** (Variable) – 对每个序列切片的起始索引
  - **length** (Variable) – 每个子序列的长度
  - **name** (str|None) – 该层的命名，可选项。 如果None, 则自动命名该层

返回：输出目标子序列

返回类型：Variable

**代码示例**

..  code-block:: python

	import numpy as np
	seqs = fluid.layers.data(name='x', shape=[10, 5],
			 dtype='float32', lod_level=1)
	offset = fluid.layers.assign(input=np.array([[0, 1]]).astype("int32"))
	length = fluid.layers.assign(input=np.array([[2, 1]]).astype("int32"))
	subseqs = fluid.layers.sequence_slice(input=seqs, offset=offset,
					      length=length)










.. _cn_api_fluid_layers_sequence_softmax:

sequence_softmax
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_softmax(input, use_cudnn=False, name=None)

该函数计算每一个序列所有时间步中的softmax激活函数。每个时间步的维度应为1。

输入张量的形状可为 :math:`[N，1]` 或者 :math:`[N]` , :math:`N` 是所有序列长度之和。

对mini-batch的第i序列：

.. math::

    Out\left ( X[lod[i]:lod[i+1]],: \right ) = \frac{exp(X[lod[i]:lod[i+1],:])}{\sum (exp(X[lod[i]:lod[i+1],:]))}

例如，对有3个序列（可变长度）的mini-batch，每个包含2，3，2时间步，其lod为[0,2,5,7]，则在 :math:`X[0:2,:],X[2:5,:],X[5:7,:]` 中进行softmax运算，并且 :math:`N` 的结果为7.

参数：
    - **input** (Variable) - 输入变量，为LoDTensor
    - **use_cudnn** (bool) - 是否用cudnn核，仅当下载cudnn库才有效。默认：False
    - **name** (str|None) - 该层名称（可选）。若设为None，则自动为该层命名。默认：None

返回：sequence_softmax的输出

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    x = fluid.layers.data(name='x', shape=[7, 1],
                 dtype='float32', lod_level=1)
    x_sequence_softmax = fluid.layers.sequence_softmax(input=x)










.. _cn_api_fluid_layers_sequence_unpad:

sequence_unpad
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_unpad(x, length, name=None)

**实现Sequence Unpad(去除序列填充值)运算**

该层从给定序列中删除padding（填充值），并且将该序列转变为未填充时的原序列作为该层的输出，并且实际长度可以在输出的LoD信息中取得。

::

    示例：

    给定输入变量 ``x`` :
        x.data = [[ 1.0,  2.0,  3.0,  4.0,  5.0],
                  [ 6.0,  7.0,  8.0,  9.0, 10.0],
                  [11.0, 12.0, 13.0, 14.0, 15.0]],

    其中包含 3 个被填充到长度为5的序列，实际长度由输入变量 ``length`` 指明：

        length.data = [[2], [3], [4]],

    则去填充（unpad）后的输出变量为：

        out.data = [[1.0, 2.0, 6.0, 7.0, 8.0, 11.0, 12.0, 13.0, 14.0]]
        out.lod = [[2, 3, 4]]



参数:
  - **x** (Variable) – 输入变量，承载着多个填充后等长的序列
  - **length** (Variable) – 变量，指明去填充后各个序列所具有的实际长度
  - **name** (str|None) – 可选项，该层名称。 若为 None, 将自动命名该层

返回：变量，承载着去填充处理后的序列

返回类型：Variable

**代码示例**

..  code-block:: python

    x = fluid.layers.data(name='x', shape=[10, 5], dtype='float32')
    len = fluid.layers.data(name='length', shape=[1], dtype='int64')
    out = fluid.layers.sequence_unpad(x=x, length=len)
    











.. _cn_api_fluid_layers_shape:

shape
-------------------------------

.. py:function:: paddle.fluid.layers.shape(input)

shape算子

获得输入张量的形状。现在只支持输入CPU的Tensor。

参数：
        - **input** （Variable）- （Tensor），输入张量。

返回：        (Tensor），输入张量的形状，形状的数据类型是int32，它将与输入张量（Tensor）在同一设备上。

返回类型：        输出（Variable）。
        
        
        








.. _cn_api_fluid_layers_sigmoid_cross_entropy_with_logits:

sigmoid_cross_entropy_with_logits
-------------------------------

.. py:function:: paddle.fluid.layers.sigmoid_cross_entropy_with_logits(x, label, name=None)

在类别不相互独立的分类任务中，该函数可以衡量按元素顺序的可能性误差。可以这么认为，为单一数据点预测标签，它们之间不是完全互斥的关系。例如，一篇新闻文章可以同时关于政治，科技，体育或者同时不包含这些内容。

逻辑loss可通过下式计算：

.. math::
    loss = -Labels * log(sigma(X)) - (1 - Labels) * log(1 - sigma(X))
 
已知:

.. math::
    sigma(X) = \frac{1}{1 + exp(-X)}

代入最开始的式子，

.. math::
    loss = X - X * Labels + log(1 + exp(-X))

为了计算稳定性，防止 :math:`exp(-X)` 溢出，当 :math:`X<0` 时，loss将采用以下公式计算:

.. math::
    loss = max(X, 0) - X * Labels + log(1 + exp(-|X|))

输入 ``X`` 和 ``label`` 都可以携带LoD信息。当输出仅采用输入 ``X`` 的LoD。



参数:
  - **x** (Variable) - (Tensor, 默认 Tensor<float>)，形为 N x D 的二维张量，N为batch大小，D为类别数目。该输入是一个由先前运算得出的logit组成的张量。logit是未标准化(unscaled)的log概率， 公式为 :math:`log(\frac{p}{1-p})` 
  - **label** (Variable) -  (Tensor, 默认 Tensor<float>) 具有和X相同类型，相同形状的二维张量。该输入张量代表了每个logit的可能标签
  - **name** (basestring|None) - 输出的名称

返回：	(Tensor, 默认Tensor<float>), 形为 N x D 的二维张量，其值代表了按元素的逻辑loss

返回类型:	Variable












.. _cn_api_fluid_layers_similarity_focus:

similarity_focus
-------------------------------

.. py:function:: paddle.fluid.layers.similarity_focus(input, axis, indexes, name=None)

**实现SimilarityFocus(相似度聚焦)运算**

通过以下三种方式，该层生成一个和输入 ``input`` 同形的mask（掩码）：

1. 根据 ``axis`` 和 ``indexes`` 提取一个三维张量，第一维为batch大小。
   例如，如果 ``axis=1, indexes=[a]`` , 将得到矩阵 T=X[:, a, :, :] 。
   该例中，如果输入X的形为 (BatchSize, A, B, C) ，则输出张量T的形为 (BatchSize, B, C) 。
2. 对于每一个索引，在输出T中找到最大值。所以同一行、同一列最多只有一个数字，这意味着如果在第i行，第j列中找到最大值，那么在相应行、列中的其他数值都将被忽略。然后再在剩余的数值中找到下一个最大值。显然，将会产生 min（B,C）个数字，并把三维相似聚焦掩码张量相应位置的元素置为1，其余则置为0。对每个索引按元素进行or运算。
3. 将这个三维相似度聚焦mask调整、适配于输入 ``input`` 的形状

请参照 `Similarity Focus Layer <http://www.aclweb.org/anthology/N16-1108>`_ 。

::
    
    例如 :

    给定四维张量 x 形为 (BatchSize, C, A, B), 其中C 为通道Channel数目，
    特征图（feature map）的形为（A,B）：
    
        x.shape = (2, 3, 2, 2)
        x.data = [[[[0.8, 0.1],
                    [0.4, 0.5]],

                   [[0.9, 0.7],
                    [0.9, 0.9]],

                   [[0.8, 0.9],
                    [0.1, 0.2]]],


                  [[[0.2, 0.5],
                    [0.3, 0.4]],

                   [[0.9, 0.7],
                    [0.8, 0.4]],

                   [[0.0, 0.2],
                    [0.4, 0.7]]]]

    给定轴: 1 (即channel轴)
    给定索引: [0]

    于是我们得到一个与输入同形的四维输出张量：
        out.shape = (2, 3, 2, 2)
        out.data = [[[[1.0, 0.0],
                      [0.0, 1.0]],

                     [[1.0, 0.0],
                      [0.0, 1.0]],

                     [[1.0, 0.0],
                      [0.0, 1.0]]],

                    [[[0.0, 1.0],
                      [1.0, 0.0]],

                     [[0.0, 1.0],
                      [1.0, 0.0]],

                     [[0.0, 1.0],
                      [1.0, 0.0]]]]



参数:
  - **input** (Variable) – 输入张量(默认类型为float)。应为一个四维张量，形为[BatchSize, A, B, C]
  - **axis** (int) – 指明要选择的轴。 可能取值为 1, 2 或 3.
  - **indexes** (list) – 指明选择维度的索引列表

返回：一个和输入张量同形、同类型的张量变量

返回类型：Variable

**代码示例**

..  code-block:: python

            data = fluid.layers.data(
              name='data', shape=[2, 3, 2, 2], dtype='float32')
            x = fluid.layers.layer_norm(input=data, axis=1, indexes=[0])












.. _cn_api_fluid_layers_slice:

slice
-------------------------------

.. py:function:: paddle.fluid.layers.slice(input, axes, starts, ends)

slice算子。

沿多个轴生成输入张量的切片。与numpy类似： https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html Slice使用 ``axes`` 、 ``starts`` 和 ``ends`` 属性来指定轴列表中每个轴的起点和终点维度，它使用此信息来对输入数据张量切片。如果向 ``starts`` 或 ``ends`` 传递负值，则表示该维度结束之前的元素数目。如果传递给 ``starts`` 或 ``end`` 的值大于n（此维度中的元素数目），则表示n。对于未知大小维度的末尾进行切片，则建议传入 ``INT_MAX`` 。如果省略轴，则将它们设置为[0，...，ndim-1]。以下示例将解释切片如何工作：

::

        案例1：给定：data=[[1,2,3,4],[5,6,7,8],] 
                     axes=[0,1] 
                     starts=[1,0] 
                     ends=[2,3] 
               则：
                     result=[[5,6,7],]

        案例2：给定：
                     data=[[1,2,3,4],[5,6,7,8],] 
                     starts=[0,1] 
                     ends=[-1,1000] 
               则：
                     result=[[2,3,4],]

参数：
        - **input** （Variable）- 提取切片的数据张量（Tensor）。
        - **axes** （List）- （list <int>）开始和结束的轴适用于。它是可选的。如果不存在，将被视为[0,1，...，len（starts）- 1]。
        - **starts** （List）- （list <int>）在轴上开始相应轴的索引。
        - **ends** （List）- （list <int>）在轴上结束相应轴的索引。

返回：        切片数据张量（Tensor）.

返回类型：        输出（Variable）。










.. _cn_api_fluid_layers_smooth_l1:

smooth_l1
-------------------------------

.. py:function:: paddle.fluid.layers.smooth_l1(x, y, inside_weight=None, outside_weight=None, sigma=None)

该layer计算变量x1和y 的smooth L1 loss，它以x和y的第一维大小作为批处理大小。对于每个实例，按元素计算smooth L1 loss，然后计算所有loss。输出变量的形状是[batch_size, 1]


参数:
        - **x** (Variable) - rank至少为2的张量。输入x的smmoth L1 loss 的op，shape为[batch_size, dim1，…],dimN]。
        - **y** (Variable) - rank至少为2的张量。与 ``x`` 形状一致的的smooth L1 loss  op目标值。
        - **inside_weight** (Variable|None) - rank至少为2的张量。这个输入是可选的，与x的形状应该相同。如果给定， ``(x - y)`` 的结果将乘以这个张量元素。
        - **outside_weight** (变量|None) - 一个rank至少为2的张量。这个输入是可选的，它的形状应该与 ``x`` 相同。如果给定，那么 smooth L1 loss 就会乘以这个张量元素。
        - **sigma** (float|None) - smooth L1 loss layer的超参数。标量，默认值为1.0。
   
返回：	smooth L1 loss, shape为 [batch_size, 1]

返回类型:  Variable    

**代码示例**

..  code-block:: python
        
    data = fluid.layers.data(name='data', shape=[128], dtype='float32')
    label = fluid.layers.data(
        name='label', shape=[100], dtype='float32')
    fc = fluid.layers.fc(input=data, size=100)
    out = fluid.layers.smooth_l1(x=fc, y=label)










.. _cn_api_fluid_layers_soft_relu:

soft_relu
-------------------------------

.. py:function:: paddle.fluid.layers.soft_relu(x, threshold=40.0, name=None)

SoftRelu 激活函数

.. math::   out=ln(1+exp(max(min(x,threshold),threshold))
 
参数:
    - **x** (variable) - SoftRelu operator的输入
    - **threshold** (FLOAT|40.0) - SoftRelu的阈值
    - **name** (str|None) - 该层的名称(可选)。如果设置为None，该层将被自动命名

**代码示例：**

.. code-block:: python

    x = fluid.layers.data(name=”x”, shape=[2,3,16,16], dtype=”float32”) 
    y = fluid.layers.soft_relu(x, threshold=20.0)








.. _cn_api_fluid_layers_softmax:

softmax
-------------------------------

.. py:function:: paddle.fluid.layers.softmax(input, use_cudnn=True, name=None)

softmax操作符的输入是任意阶的张量，输出张量和输入张量的维度相同。

首先逻辑上将输入张量压平至二维矩阵。矩阵的第二维（行数）和输入张量的最后一维相同。第一维（列数）
是输入张量除最后一维之外的所有维的产物。对矩阵的每一行来说,softmax操作将K维(K是矩阵的宽度,也就是输入张量的维度)任意实际值，压缩成K维取值为[0,1]之间的向量，压缩后k个值的和为1。


softmax操作符计算k维向量输入中所有其他维的指数和指数值的累加和。维的指数比例和所有其他维的指数值之和作为softmax操作符的输出。

对矩阵中的每行i和每列j有：

.. math::

    Out[i,j] = \frac{exp(X[i,j])}{\sum{j}_exp(X[i,j])}

参数：
    - **input** (Variable) - 输入变量
    - **use_cudnn** (bool) - 是否用cudnn核，只有在cudnn库安装时有效
    - **name** (str|None) - 该层名称（可选）。若为空，则自动为该层命名。默认：None

返回： softmax输出

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    fc = fluid.layers.fc(input=x, size=10)
    softmax = fluid.layers.softmax(input=fc)









.. _cn_api_fluid_layers_softmax_with_cross_entropy:

softmax_with_cross_entropy
-------------------------------

.. py:function:: paddle.fluid.layers.softmax_with_cross_entropy(logits, label, soft_label=False, ignore_index=-100, numeric_stable_mode=False, return_softmax=False)

使用softmax的交叉熵在输出层已被广泛使用。该函数计算输入张量每一行的softmax标准化值，而后计算交叉熵。通过此种方式，可以得到更具数字稳定性的梯度值。

因为该运算是在内部进行logit上的softmax运算，所以它需要未标准化（unscaled）的logit。该运算不应该对softmax运算的输出进行操作，否则会得出错误结果。

当 ``soft_label`` 为 ``False`` 时，该运算接受互斥的硬标签，batch中的每一个样本都以为1的概率分类到一个类别中，并且仅有一个标签。

涉及到的等式如下:

1.硬标签，即 one-hot label, 每个样本仅可分到一个类别

.. math::
   loss_j =  -\text{logit}_{label_j} +\log\left(\sum_{i=0}^{K}\exp(\text{logit}_i)\right), j = 1,..., K

2.软标签，每个样本可能被分配至多个类别中

.. math::
   loss_j =  -\sum_{i=0}^{K}\text{label}_i\left(\text{logit}_i - \log\left(\sum_{i=0}^{K}\exp(\text{logit}_i)\right)\right), j = 1,...,K

3.如果 ``numeric_stable_mode`` 为真，在通过softmax和标签计算交叉熵损失前， softmax 首先经由下式计算得出：

.. math::
    max_j           &= \max_{i=0}^{K}{\text{logit}_i} \\
    log\_max\_sum_j &= \log\sum_{i=0}^{K}\exp(logit_i - max_j)\\
    softmax_j &= \exp(logit_j - max_j - {log\_max\_sum}_j)


参数:
  - **logits** (Variable) - 未标准化(unscaled)的log概率,一个形为 N X K 的二维张量。 N是batch大小，K是类别总数
  - **label** (Variable) - 2-D 张量，代表了正确标注（ground truth）, 如果 ``soft_label`` 为  False，则该参数是一个形为 N X 1 的Tensor<int64> 。如果 ``soft_label`` 为 True，它是 Tensor<float/double> ，形为 N X K
  - **soft_label** (bool) - 是否将输入标签当作软标签。默认为False
  - **ignore_index** (int) - 指明要无视的目标值，使之不对输入梯度有贡献。仅在 ``soft_label`` 为False时有效，默认为-100 
  - **numeric_stable_mode** (bool) – 标志位，指明是否使用一个具有更佳数学稳定性的算法。仅在 ``soft_label`` 为 False的GPU模式下生效. 若 ``soft_label`` 为 True 或者执行场所为CPU, 算法一直具有数学稳定性。 注意使用稳定算法时速度可能会变慢。默认为 False
  - **return_softmax** (bool) – 标志位，指明是否额外返回一个softmax值， 同时返回交叉熵计算结果。默认为False

返回: 
  - 如果 ``return_softmax`` 为 False， 则返回交叉熵，一个形为[N x 1]的二维张量
  - 如果 ``return_softmax`` 为 True，则返回元组 (loss, softmax) ，其中交叉熵loss为形为[N x 1]的二维张量，softmax为[N x K]的二维张量

返回类型:变量或者两个变量组成的元组


**代码示例**

..  code-block:: python

	data = fluid.layers.data(name='data', shape=[128], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        fc = fluid.layers.fc(input=data, size=100)
        out = fluid.layers.softmax_with_cross_entropy(
        logits=fc, label=label)
      
      








.. _cn_api_fluid_layers_space_to_depth:

space_to_depth
-------------------------------

.. py:function:: paddle.fluid.layers.space_to_depth(x, blocksize, name=None)

给该函数一个 ``blocksize`` 值，可以对形为[batch, channel, height, width]的输入LoD张量进行space_to_depth（广度至深度）运算。

该运算对成块的空间数据进行重组，形成深度。确切地说，该运算输出一个输入LoD张量的拷贝，其高度，宽度维度上的值移动至通道维度上。

``blocksize`` 参数指明了数据块大小。

重组时，依据 ``blocksize`` , 生成形为 :math:`[batch, channel * blocksize * blocksize, height/blocksize, width/blocksize]` 的输出：

该运算适用于在卷积间重放缩激励函数，并保持所有的数据。

 - 在各位置上，不重叠的，大小为 :math:`block\_size * block\_size` 的块重组入深度depth
 - 输出张量的深度为 :math:`block\_size * block\_size * input\_channel`
 - 输入各个块中的Y,X坐标变为输出张量通道索引的高序部位 
 - channel可以被blocksize的平方整除
 - 高度，宽度可以被blocksize整除

参数:
  - **x** (variable) – 输入LoD张量
  - **blocksize** (variable) – 在每个特征图上选择元素时采用的块大小，应该 > 2

返回：输出LoD tensor

返回类型：Variable

抛出异常： 
  - ``TypeError`` - ``blocksize`` 必须是long类型

**代码示例**

..  code-block:: python

    data = fluid.layers.data(
        name='data', shape=[1, 4, 2, 2], dtype='float32')
    space_to_depthed = fluid.layers.space_to_depth(
        x=data, blocksize=2)










.. _cn_api_fluid_layers_split:

split
-------------------------------

.. py:function:: paddle.fluid.layers.split(input,num_or_sections,dim=-1,name=None)

将输入张量分解成多个子张量

参数：
    - **input** (Variable)-输入变量，类型为Tensor或者LoDTensor
    - **num_or_sections** (int|list)-如果num_or_sections是整数，则表示张量平均划分为的相同大小子张量的数量。如果num_or_sections是一列整数，列表的长度代表子张量的数量，整数依次代表子张量的dim维度的大小
    - **dim** (int)-将要划分的维。如果dim<0,划分的维为rank(input)+dim
    - **name** (str|None)-该层名称（可选）。如果设置为空，则自动为该层命名

返回：一列分割张量

返回类型：列表(Variable)

**代码示例**：

.. code-block:: python

    # x是维为[3,9,5]的张量：
    x0, x1, x2 = fluid.layers.split(x, num_or_sections=3, dim=1)
    x0.shape  # [3, 3, 5]
    x1.shape  # [3, 3, 5]
    x2.shape  # [3, 3, 5]
    x0, x1, x2 = fluid.layers.split(
        x, num_or_sections=[2, 3, 4], dim=1)
    x0.shape  # [3, 2, 5]
    x1.shape  # [3, 3, 5]
    x2.shape  # [3, 4, 5]









.. _cn_api_fluid_layers_square_error_cost:

square_error_cost 
-------------------------------

.. py:function:: paddle.fluid.layers.square_error_cost(input,label)

方差估计层（Square error cost layer）

该层接受输入预测值和目标值，并返回方差估计

对于预测值X和目标值Y，公式为：

.. math::

    Out = (X-Y)^{2}

在以上等式中：
    - **X** : 输入预测值，张量（Tensor)
    - **Y** : 输入目标值，张量（Tensor）
    - **Out** : 输出值，维度和X的相同

参数：
    - **input** (Variable) - 输入张量（Tensor），带有预测值
    - **label** (Variable) - 标签张量（Tensor），带有目标值

返回：张量变量，存储输入张量和标签张量的方差

返回类型：变量（Variable）

**代码示例**：

 .. code-block:: python

    y = layers.data(name='y', shape=[1], dtype='float32')
    y_predict = layers.data(name='y_predict', shape=[1], dtype='float32')
    cost = layers.square_error_cost(input=y_predict, label=y)









.. _cn_api_fluid_layers_squeeze:

squeeze 
-------------------------------

.. py:function:: paddle.fluid.layers.squeeze(input, axes, name=None)

向张量维度中移除单维输入。传入用于压缩的轴。如果未提供轴，所有的单一维度将从维中移除。如果选择的轴的形状条目不等于1，则报错。

::


    例如：

    例1：
        给定
            X.shape = (1,3,1,5)
            axes = [0]
        得到
            Out.shape = (3,1,5)
    例2：
        给定
            X.shape = (1,3,1,5)
            axes = []
        得到
            Out.shape = (3,5)

参数：
        - **input** (Variable)-将要压缩的输入变量
        - **axes** (list)-一列整数，代表压缩的维
        - **name** (str|None)-该层名称

返回：输出压缩的变量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    x = layers.data(name='x', shape=[5, 1, 10])
    y = layers.sequeeze(input=x, axes=[1])      









.. _cn_api_fluid_layers_stack:

stack
-------------------------------

.. py:function:: paddle.fluid.layers.stack(x, axis=0)

实现了stack层。

沿 ``axis`` 轴，该层对输入 ``x`` 进行stack运算。

输入 x 可以是单个变量, 或是多个变量组成的列表或元组。如果 x 是一个列表或元组, 那么这些变量必须同形。 假设每个输入的形都为 :math:`[d_0,d_1,...,d_{n−1}]` , 则输出变量的形为 :math:`[d_0,d_1,...,d_{axis}=len(x),...,d_{n−1}]` 。 如果 ``axis`` < 0, 则将其取代为 :math:`axis+rank(x[0])+1` 。 如果 ``axis`` 为 None, 则认为它是 0。

参数:	
  - **x** (Variable|list(Variable)|tuple(Variable)) – 输入变量
  - **axis** (int|None) – 对输入进行stack运算所在的轴

返回: 经stack运算后的变量

返回类型: Variable











.. _cn_api_fluid_layers_stanh:

stanh
-------------------------------

.. py:function:: paddle.fluid.layers.stanh(x, scale_a=0.6666666666666666, scale_b=1.7159, name=None)

STanh 激活算子（STanh Activation Operator.）

.. math::      
          \\out=b*\frac{e^{a*x}-e^{-a*x}}{e^{a*x}+e^{-a*x}}\\

参数：
    - **x** (Variable) - STanh operator的输入
    - **scale_a** (FLOAT|2.0 / 3.0) - 输入的a的缩放参数
    - **scale_b** (FLOAT|1.7159) - b的缩放参数
    - **name** (str|None) - 这个层的名称(可选)。如果设置为None，该层将被自动命名。

返回: STanh操作符的输出

返回类型: 输出(Variable)

**代码示例：**

.. code-block:: python

    x = fluid.layers.data(name="x", shape=[3,10,32,32], dtype="float32")
    y = fluid.layers.stanh(x, scale_a=0.67, scale_b=1.72)







.. _cn_api_fluid_layers_sum:

sum
-------------------------------

.. py:function:: paddle.fluid.layers.sum(x)

sum算子。

该运算符对输入张量求和。所有输入都可以携带LoD（详细程度）信息，但是输出仅与第一个输入共享LoD信息。

参数：
        - **x** （Variable）- （vector <Tensor>）sum运算符的输入张量（Tensor）。

返回:        (Tensor）求和算子的输出张量。

返回类型：        Variable










.. _cn_api_fluid_layers_swish:

swish
-------------------------------

.. py:function:: paddle.fluid.layers.swish(x, beta=1.0, name=None)

Swish 激活函数

.. math::   
         \\out = \frac{x}{e^(1+betax)}\\

参数：
    - **x** (Variable) -  Swishoperator的输入
    - **beta** (浮点|1.0) - Swish operator 的常量beta
    - **name** (str|None) - 这个层的名称(可选)。如果设置为None，该层将被自动命名。

返回: Swish operator 的输出

返回类型: output(Variable)


**代码示例：**

.. code-block:: python

  x = fluid.layers.data(name="x", shape=[3,10,32,32], dtype="float32")
  y = fluid.layers.swish(x, beta=2.0)









.. _cn_api_fluid_layers_topk:

topk
-------------------------------
.. py:function:: paddle.fluid.layers.topk(input, k, name=None)

这个运算符用于查找最后一维的前k个最大项，返回它们的值和索引。

如果输入是（1-D Tensor），则找到向量的前k最大项，并以向量的形式输出前k最大项的值和索引。values[j]是输入中第j最大项，其索引为indices[j]。
如果输入是更高阶的张量，则该operator会基于最后一维计算前k项

例如：

.. code-block:: text


    如果:
        input = [[5, 4, 2, 3],
                [9, 7, 10, 25],
                [6, 2, 10, 1]]
        k = 2

    则:
        第一个输入:
        values = [[5, 4],
                [10, 25],
                [6, 10]]

        第二个输入:
        indices = [[0, 1],
                [2, 3],
                [0, 2]]

参数：
    - **input** (Variable)-输入变量可以是一个向量或者更高阶的张量
    - **k** (int)-在输入最后一纬中寻找的前项数目
    - **name** (str|None)-该层名称（可选）。如果设为空，则自动为该层命名。默认为空

返回：含有两个元素的元组。元素都是变量。第一个元素是最后维切片的前k项。第二个元素是输入最后维里值索引

返回类型：元组[变量]

提示：抛出异常-如果k<1或者k不小于输入的最后维

**代码示例**：

.. code-block:: python 

    top5_values, top5_indices = layers.topk(input, k=5)









.. _cn_api_fluid_layers_transpose:

transpose
-------------------------------

.. py:function:: paddle.fluid.layers.transpose(x,perm,name=None)

根据perm对输入矩阵维度进行重排。

返回张量（tensor）的第i维对应输入维度矩阵的perm[i]。

参数：
    - **x** (Variable) - 输入张量（Tensor)
    - **perm** (list) - 输入维度矩阵的转置
    - **name** (str) - 该层名称（可选）

返回： 转置后的张量（Tensor）

返回类型：变量（Variable）

**代码示例**:

.. code-block:: python

    # 请使用 append_batch_size=False 来避免
    # 在数据张量中添加多余的batch大小维度
    x = fluid.layers.data(name='x', shape=[5, 10, 15],
                    dtype='float32', append_batch_size=False)
    x_transposed = layers.transpose(x, perm=[1, 0, 2])









.. _cn_api_fluid_layers_uniform_random_batch_size_like:

uniform_random_batch_size_like
-------------------------------

.. py:function:: paddle.fluid.layers.uniform_random_batch_size_like(input, shape, dtype='float32', input_dim_idx=0, output_dim_idx=0, min=-1.0, max=1.0, seed=0)

uniform_random_batch_size_like算子。

此运算符使用与输入张量（Tensor）相同的batch_size初始化张量（Tensor），并使用从均匀分布中采样的随机值。

参数：
        - **input** （Variable）- 其input_dim_idx'th维度指定batch_size的张量（Tensor）。
        - **shape** （元组|列表）- 输出的形状。
        - **input_dim_idx** （Int）- 默认值0.输入批量大小维度的索引。
        - **output_dim_idx** （Int）- 默认值0.输出批量大小维度的索引。
        - **min** （Float）- （默认 1.0）均匀随机的最小值。
        - **max** （Float）- （默认 1.0）均匀随机的最大值。
        - **seed** （Int）- （int，default 0）用于生成样本的随机种子。0表示使用系统生成的种子。注意如果seed不为0，则此运算符将始终每次生成相同的随机数。
        - **dtype** （np.dtype | core.VarDesc.VarType | str） - 数据类型：float32，float_16，int等。

返回:        指定形状的张量（Tensor）将使用指定值填充。

返回类型:        Variable









.. _cn_api_fluid_layers_unsqueeze:

unsqueeze
-------------------------------

.. py:function:: paddle.fluid.layers.unsqueeze(input, axes, name=None)

向张量shape中插入单维函数。获取一个必需axes值，用来插入维度列表。输出张量显示轴的维度索引值。

比如：
    给定一个张量，例如维度为[3,4,5]的张量，轴为[0,4]的未压缩张量，维度为[1,3,4,5,1]

参数：
    - **input** (Variable)- 未压缩的输入变量
    - **axes** (list)- 一列整数，代表要插入的维数
    - **name** (str|None) - 该层名称

返回：输出未压缩变量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    x = layers.data(name='x', shape=[5, 10])
    y = layers.unsequeeze(input=x, axes=[1])










.. _cn_api_fluid_layers_unstack:

unstack
-------------------------------

.. py:function:: paddle.fluid.layers.unstack(x, axis=0, num=None)

实现了unstack层。

沿 ``axis`` 轴，该层对输入 ``x`` 进行unstack运算。

如果 ``axis`` <0，则将其以 :math:`axis+rank(x)` 代之。

如果 ``num`` 为 None，则它可以从 ``x.shape[axis]`` 中推断而来。

如果 ``x.shape[axis]`` <= 0或者Unknown, 则抛出异常 ``ValueError`` 。 

参数:	
  - **x** (Variable|list(Variable)|tuple(Variable)) – 输入变量
  - **axis** (int|None) – 对输入进行unstack运算所在的轴
  - **num** (int|None) - 输出变量的数目

返回: 经unstack运算后的变量

返回类型: Variable
  










.. _cn_api_fluid_layers_warpctc:

warpctc
-------------------------------

.. py:function:: paddle.fluid.layers.warpctc(input, label, blank=0, norm_by_times=False, use_cudnn=False))

该操作符集成了 `开源Warp-CTC库 <https://github.com/baidu-research/warp-ctc>`_ ，计算基于神经网络的时序类分类（CTC）损失。原生softmax激活函数集成到Wrap-CTC库中，操作符也可称作含CTC的softmax，将输入张量每一行的值正则化。

参数：
    - **input** （Variable） - 变长序列的非尺度化概率，是一个含LoD信息的二维张量。shape为[Lp，num_classes+1]，Lp是所有输出序列长度之和，num_classes是实际类别数。（不包括空白标签）
    - **label** (Variable） - 变长序列中正确标记的数据，是一个含LoD信息的二维张量。shape为[Lg，1]，Lg是所有标签长度之和
    - **blank** （int，默认0） - 基于神经网络的时序类分类（CTC）损失的空白标签索引，在半开区间间隔内[0，num_classes+1]
    - **norm_by_times** （bool，默认false） - 是否利用时间步长（即序列长度）的数量对梯度进行正则化。如果warpctc层后面跟着mean_op则无需对梯度正则化。
    - **use_cudnn** (bool, 默认false) - 是否使用cudnn

返回：基于神经网络的时序类分类（CTC）损失，是一个shape为[batch_size，1]的二维张量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    label = fluid.layers.data(shape=[11, 8], dtype='float32', lod_level=1)
    predict = fluid.layers.data(shape=[11, 1], dtype='float32')
    cost = fluid.layers.warpctc(input=predict, label=label)











============
 ops 
============


.. _cn_api_fluid_layers_abs:

abs
-------------------------------

.. py:function:: paddle.fluid.layers.abs(x, name=None)


参数:
    - **x** - abs运算符的输入 
    - **use_mkldnn** (bool) - （默认为false）仅在 ``mkldnn`` 内核中使用


返回：        Abs运算符的输出。











.. _cn_api_fluid_layers_ceil:

ceil
-------------------------------

.. py:function:: paddle.fluid.layers.ceil(x, name=None)


参数:
    - **x** - Ceil运算符的输入 
    - **use_mkldnn** (bool) - （默认为false）仅在 ``mkldnn`` 内核中使用

返回：        Ceil运算符的输出。
        
        








.. _cn_api_fluid_layers_cos:

cos
-------------------------------

.. py:function:: paddle.fluid.layers.cos(x, name=None)



参数:
    - **x** - cos运算符的输入 
    - **use_mkldnn** (bool) - （默认为false）仅在 ``mkldnn`` 内核中使用

返回：        Cos运算符的输出










.. _cn_api_fluid_layers_cumsum:

cumsum
-------------------------------

.. py:function:: paddle.fluid.layers.cumsum(x,axis=None,exclusive=None,reverse=None

沿给定轴的元素的累加和。默认结果的第一个元素和输入的第一个元素一致。如果exlusive为真，结果的第一个元素则为0。

参数：
    - **x** -累加操作符的输入
    - **axis** (INT)-需要累加的维。-1代表最后一维。[默认 -1]。
    - **exclusive** (BOOLEAN)-是否执行exclusive累加。[默认false]。
    - **reverse** (BOOLEAN)-若为true,则以相反顺序执行累加。[默认 false]。

返回：累加器的输出

**代码示例**：

.. code-block:: python

    data = fluid.layers.data(name="input", shape=[32, 784])
    result = fluid.layers.cumsum(data, axis=0)









.. _cn_api_fluid_layers_exp:

exp
-------------------------------

.. py:function:: paddle.fluid.layers.exp(x, name=None)
       

参数:
    - **x** - Exp运算符的输入 
    - **use_mkldnn** (bool) - （默认为false）仅在 ``mkldnn`` 内核中使用

返回：       Exp算子的输出










.. _cn_api_fluid_layers_floor:

floor
-------------------------------

.. py:function:: paddle.fluid.layers.floor(x, name=None)



参数:
    - **x** - Floor运算符的输入 
    - **use_mkldnn** (bool) - （默认为false）仅在 ``mkldnn`` 内核中使用

返回：        Floor运算符的输出。











.. _cn_api_fluid_layers_hard_shrink:

hard_shrink
-------------------------------

.. py:function:: paddle.fluid.layers.hard_shrink(x,threshold=None)

HardShrink激活函数(HardShrink activation operator)


.. math::
	
	out = \begin{cases}
        x, \text{if } x > \lambda \\
        x, \text{if } x < -\lambda \\
        0,  \text{otherwise}
      \end{cases}

参数：
    - **x** - HardShrink激活函数的输入
    - **threshold** (FLOAT)-HardShrink激活函数的threshold值。[默认：0.5]

返回：HardShrink激活函数的输出

**代码示例**：

    .. code-block:: python

        data = fluid.layers.data(name="input", shape=[784])
        result = fluid.layers.hard_shrink(x=data, threshold=0.3)    









.. _cn_api_fluid_layers_logsigmoid:

logsigmoid
-------------------------------

.. py:function:: paddle.fluid.layers.logsigmoid(x, name=None)
        


参数:
    - **x** - LogSigmoid运算符的输入 
    - **use_mkldnn** (bool) - （默认为False）仅在 ``mkldnn`` 内核中使用

返回：        LogSigmoid运算符的输出










.. _cn_api_fluid_layers_reciprocal:

reciprocal
-------------------------------

.. py:function:: paddle.fluid.layers.reciprocal(x, name=None)


参数:
    - **x** - Ceil运算符的输入 
    - **use_mkldnn** (bool) - （默认为false）仅在 ``mkldnn`` 内核中使用

返回：        Reciprocal运算符的输出。        



        
        








.. _cn_api_fluid_layers_round:

round
-------------------------------

.. py:function:: paddle.fluid.layers.round(x, name=None)


参数:
    - **x** - Ceil运算符的输入 
    - **use_mkldnn** (bool) - （默认为false）仅在 ``mkldnn`` 内核中使用

返回：        Round运算符的输出。
        
        








.. _cn_api_fluid_layers_sigmoid:

sigmoid
-------------------------------

.. py:function:: paddle.fluid.layers.sigmoid(x, name=None)
     


参数:
    - **x** - Sigmoid运算符的输入 
    - **use_mkldnn** (bool) - （默认为false）仅在 ``mkldnn`` 内核中使用


返回：     Sigmoid运算输出.


 







.. _cn_api_fluid_layers_sin:

sin
-------------------------------

.. py:function:: paddle.fluid.layers.sin(x, name=None)


参数:
    - **x** - sin运算符的输入 
    - **use_mkldnn** (bool) - （默认为false）仅在 ``mkldnn`` 内核中使用

返回：        Sin运算符的输出。











.. _cn_api_fluid_layers_softplus:

softplus
-------------------------------

.. py:function:: paddle.fluid.layers.softplus(x,name=None)

参数：
    - **x** : Softplus操作符的输入
    - **use_mkldnn** (bool, 默认false) - 仅在mkldnn核中使用

返回：Softplus操作后的结果









.. _cn_api_fluid_layers_softshrink:

softshrink
-------------------------------

.. py:function:: paddle.fluid.layers.softshrink(x, name=None)       

Softshrink激活算子

.. math::
        out = \begin{cases}
                    x - \lambda, \text{if } x > \lambda \\
                    x + \lambda, \text{if } x < -\lambda \\
                    0,  \text{otherwise}
              \end{cases}
       
参数：
        - **x** - Softshrink算子的输入 
        - **lambda** （FLOAT）- 非负偏移量。

返回：       Softshrink运算符的输出










.. _cn_api_fluid_layers_softsign:

softsign
-------------------------------

.. py:function:: Paddle.fluid.layers.softsign(x,name=None)

参数：
    - **x** : Softsign操作符的输入
    - **use_mkldnn** (bool, 默认false) - 仅在mkldnn核中使用

返回：Softsign操作后的结果









.. _cn_api_fluid_layers_sqrt:

sqrt
-------------------------------

.. py:function:: paddle.fluid.layers.sqrt(x, name=None)


参数:
    - **x** - Sqrt运算符的输入 
    - **use_mkldnn** (bool) - （默认为false）仅在 ``mkldnn`` 内核中使用

返回：       Sqrt算子的输出。











.. _cn_api_fluid_layers_square:

square
-------------------------------

.. py:function:: paddle.fluid.layers.square(x,name=None)

参数:
    - **x** : 平方操作符的输入
    - **use_mkldnn** (bool, 默认false) 仅在mkldnn核中使用

返回：平方后的结果









.. _cn_api_fluid_layers_tanh:

tanh
-------------------------------

.. py:function:: paddle.fluid.layers.tanh(x, name=None)
        


参数:
    - **x** - Tanh运算符的输入  
    - **use_mkldnn** (bool) - （默认为false）仅在 ``mkldnn`` 内核中使用


返回：     Tanh算子的输出。











.. _cn_api_fluid_layers_tanh_shrink:

tanh_shrink
-------------------------------

.. py:function:: paddle.fluid.layers.tanh_shrink(x, name=None)


参数:
    - **x** - TanhShrink运算符的输入 
    - **use_mkldnn** (bool)- （默认为false）仅在 ``mkldnn`` 内核中使用

返回：     tanh_shrink算子的输出









.. _cn_api_fluid_layers_thresholded_relu:

thresholded_relu
-------------------------------

.. py:function:: paddle.fluid.layers.thresholded_relu(x,threshold=None)

    ThresholdedRelu激活函数

    .. math::

        out = \left\{\begin{matrix}
            x, if&x > threshold\\ 
            0, &otherwise 
            \end{matrix}\right.

参数：
        - **x** -ThresholdedRelu激活函数的输入
        - **threshold** (FLOAT)-激活函数threshold的位置。[默认1.0]。
    
    返回：ThresholdedRelu激活函数的输出

    **代码示例**：

    .. code-block:: python

        data = fluid.layers.data(name="input", shape=[1])
        result = fluid.layers.thresholded_relu(data, threshold=0.4)









.. _cn_api_fluid_layers_uniform_random:

uniform_random
-------------------------------

.. py:function:: Paddle.fluid.layers.uniform_random(shape,dtype=None,min=None,max=None,seed=None)
该操作符初始化一个张量，该张量的值是从正太分布中抽样的随机值

参数：
    - **shape** (LONGS)-输出张量的维
    - **min** (FLOAT)-均匀随机分布的最小值。[默认 -1.0]
    - **max** (FLOAT)-均匀随机分布的最大值。[默认 1.0]
    - **seed** (INT)-随机种子，用于生成样本。0表示使用系统生成的种子。注意如果种子不为0，该操作符每次都生成同样的随机数。[默认 0]
    - **dtype** (INT)-输出张量数据类型。[默认5(FP32)]

返回：正态随机操作符的输出张量

**代码示例**：

.. code-block:: python

    result = fluid.layers.uniform_random(shape=[32, 784])











============
 tensor 
============


.. _cn_api_fluid_layers_argmax:

argmax
-------------------------------

.. py:function:: paddle.fluid.layers argmin(x,axis=0)
    
    **argmax**
    
    该功能计算输入张量元素中最大元素的索引，张量的元素在提供的轴上。

    参数：
        - **x** (Variable)-用于计算最大元素索引的输入
        - **axis** (int)-用于计算索引的轴
    
    返回：存储在输出中的张量

    返回类型：变量（Variable）

    **代码示例**：

    .. code-block:: python

        out = fluid.layers.argmax(x=in, axis=0)
        out = fluid.layers.argmax(x=in, axis=-1)









.. _cn_api_fluid_layers_argmin:

argmin
-------------------------------

.. py:function:: paddle.fluid.layers argmin(x,axis=0)
    
    **argmin**
    
    该功能计算输入张量元素中最小元素的索引，张量元素在提供的轴上。

    参数：
        - **x** (Variable)-计算最小元素索引的输入
        - **axis** (int)-计算索引的轴
    
    返回：存储在输出中的张量

    返回类型：变量（Variable）

    **代码示例**：

    .. code-block:: python

        out = fluid.layers.argmin(x=in, axis=0)
        out = fluid.layers.argmin(x=in, axis=-1)
    








.. _cn_api_fluid_layers_argsort:

argsort
-------------------------------

.. py:function:: paddle.fluid.layers argsort(input,axis=-1,name=None)

对给定轴上的输入变量进行排序，输出排序好的数据和相应的索引，其维度和输入相同

.. code-block:: text

    例如： 
	给定 input 并指定 axis=-1

        input = [[0.15849551, 0.45865775, 0.8563702 ],
                [0.12070083, 0.28766365, 0.18776911]],

    	执行argsort操作后，得到排序数据：

        out = [[0.15849551, 0.45865775, 0.8563702 ],
            [0.12070083, 0.18776911, 0.28766365]],
	
	根据指定axis排序后的数据indices变为:

        indices = [[0, 1, 2],
                [0, 2, 1]]

参数：
    - **input** (Variable)-用于排序的输入变量
    - **axis** (int)-含有用于排序输入变量的轴。当axis<0,实际的轴为axis+rank(input)。默认为-1，即最后一维。
    - **name** (str|None)-（可选）该层名称。如果设为空，则自动为该层命名。

返回：含有已排序的数据和索引

返回类型：元组

**代码示例**：

.. code-block:: python

    input = fluid.layers.data(data=[2, 3])
    out, indices = fluid.layers.argsort(input, axis=0)









.. _cn_api_fluid_layers_assign:

assign
-------------------------------

.. py:function:: paddle.fluid.layers.assign(input,output=None)

**Assign**

该功能将输入变量复制到输出变量

参数：
    - **input** (Variable|numpy.ndarray)-源变量
    - **output** (Variable|None)-目标变量

返回：作为输出的目标变量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    out = fluid.layers.create_tensor(dtype='float32')
    hidden = fluid.layers.fc(input=data, size=10)
    fluid.layers.assign(hidden, out)









.. _cn_api_fluid_layers_cast:

cast 
-------------------------------

.. py:function:: paddle.fluid.layers.cast(x,dtype)

该层传入变量x,并用x.dtype将x转换成dtype类型，作为输出。

参数：
    - **x** (Variable)-转换函数的输入变量
    - **dtype** (np.dtype|core.VarDesc.VarType|str)-输出变量的数据类型

返回：转换后的输出变量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    data = fluid.layers.data(name='x', shape=[13], dtype='float32')
    result = fluid.layers.cast(x=data, dtype='float64')









.. _cn_api_fluid_layers_concat:

concat
-------------------------------

.. py:function:: paddle.fluid.layers.concat(input,axis=0,name=None)

**Concat** 

这个函数将输入连接在前面提到的轴上，并将其作为输出返回。

参数：
    - **input** (list)-将要联结的张量列表
    - **axis** (int)-数据类型为整型的轴，其上的张量将被联结
    - **name** (str|None)-该层名称（可选）。如果设为空，则自动为该层命名。

返回：输出的联结变量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    out = fluid.layers.concat(input=[Efirst, Esecond, Ethird, Efourth])









.. _cn_api_fluid_layers_create_global_var:

create_global_var
-------------------------------

.. py:function:: paddle.fluid.layers.create_global_var(shape,value,dtype,persistable=False,force_cpu=False,name=None)

在全局块中创建一个新的带有值的张量。

参数：
    - **shape** (list[int])-变量的维度
    - **value** (float)-变量的值。填充新创建的变量
    - **dtype** (string)-变量的数据类型
    - **persistable** (bool)-如果是永久变量。默认：False
    - **force_cpu** (bool)-将该变量压入CPU。默认：False
    - **name** (str|None)-变量名。如果设为空，则自动创建变量名。默认：None.

返回：创建的变量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    var = fluid.create_global_var(shape=[2,3], value=1.0, dtype='float32',
                     persistable=True, force_cpu=True, name='new_var')









.. _cn_api_fluid_layers_create_parameter:

create_parameter
-------------------------------

.. py:function:: paddle.fluid.layers.create_parameter(shape,dtype,name=None,attr=None,is_bias=False,default_initializer=None)

创建一个参数。该参数是一个可学习的变量，拥有梯度并且可优化。

注：这是一个低级别的API。如果您希望自己创建新的op，这个API将非常有用，无需使用layers。

参数：
    - **shape** (list[int])-参数的维度
    - **dtype** (string)-参数的元素类型
    - **attr** (ParamAttr)-参数的属性
    - **is_bias** (bool)-当default_initializer为空，该值会对选择哪个默认初始化程序产生影响。如果is_bias为真，则使用initializer.Constant(0.0)，否则使用Xavier()。
    - **default_initializer** (Initializer)-参数的初始化程序

返回：创建的参数

**代码示例**：

.. code-block:: python

    W = fluid.layers.create_parameter(shape=[784, 200], dtype='float32')
    data = fluid.layers.data(name="img", shape=[64, 784], append_batch_size=False)
    hidden = fluid.layers.matmul(x=data, y=W)









.. _cn_api_fluid_layers_create_tensor:

create_tensor
-------------------------------

.. py:function:: paddle.fluid.layers.create_tensor(dtype,name=None,persistable=False)

创建一个变量，存储数据类型为dtype的LoDTensor。

参数：
    - **dtype** (string)-“float32”|“int32”|..., 创建张量的数据类型。
    - **name** (string)-创建张量的名称。如果未设置，则随机取一个唯一的名称。
    - **persistable** (bool)-为创建张量设置的永久标记

返回：存储在创建张量中的张量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    tensor = fluid.layers.create_tensor(dtype='float32')










.. _cn_api_fluid_layers_fill_constant:

fill_constant
-------------------------------

.. py:function:: paddle.fluid.layers fill_constant(shape,dtype,value,force_cpu=False,out=None)

**fill_constant**

该功能创建一个张量，具体含有shape,dtype和batch尺寸。并用值中提供的常量初始化该张量。

创建张量的属性stop_gradient设为True。

参数：
    - **shape** (tuple|list|None)-输出张量的维
    - **dtype** (np.dtype|core.VarDesc.VarType|str)-输出张量的数据类型
    - **value** (float)-用于初始化输出张量的常量值
    - **out** (Variable)-输出张量
    - **force_cpu** (True|False)-若设为true,数据必须在CPU上

返回：存储在输出中的张量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    data = fluid.layers.fill_constant(shape=[1], value=0, dtype='int64')









.. _cn_api_fluid_layers_fill_constant_batch_size_like:

fill_constant_batch_size_like
-------------------------------

.. py:function:: paddle.fluid.layers.fill_constant_batch_size_like(input,shape,dtype,value,input_dim_idx=0,output_dim_idx=0)

该功能创建一个张量，具体含有shape,dtype和batch尺寸。并用值中提供的常量初始化该张量。该批尺寸从输入张量中获取。它还将stop_gradient设置为True.

参数：
    - **input** (Variable)-张量，其input_dim_idx个维具体指示batch_size
    - **shape** (INTS)-输出的维
    - **dtype** (INT)-可以为numpy.dtype。输出数据类型。默认为float32
    - **value** (FLOAT)-默认为0.将要被填充的值
    - **input_dim_idx** (INT)-默认为0.输入批尺寸维的索引
    - **output_dim_idx** (INT)-默认为0.输出批尺寸维的索引

返回：具体维的张量填充有具体值

**代码示例**：

.. code-block:: python

    data = fluid.layers.fill_constant_batch_size_like(
                input=like, shape=[1], value=0, dtype='int64')










.. _cn_api_fluid_layers_has_inf:

has_inf
-------------------------------

.. py:function:: paddle.fluid.layers.has_inf(x)

测试x是否包括一个无穷数

参数：
	- **x(variable)** - 用于被检查的Tensor/LoDTensor

返回：
  tensor变量存储输出值，包含一个bool型数值















.. _cn_api_fluid_layers_has_nan:

has_nan
-------------------------------

.. py:function:: paddle.fluid.layers.has_nan(x)

测试x是否包含NAN

参数：
	- **x(variable)** - 用于被检查的Tensor/LoDTensor

返回：
  tensor变量存储输出值，包含一个bool型数值









.. _cn_api_fluid_layers_isfinite:

isfinite
-------------------------------

.. py:function:: paddle.fluid.layers.isfinite(x)

测试x是否包含无穷大/NAN值，如果所有元素都是有穷数，返回Ture,否则返回False

参数：
  - **x(variable)** - 用于被检查的Tensor/LoDTensor

返回:
  Variable: tensor变量存储输出值，包含一个bool型数值










.. _cn_api_fluid_layers_ones:

ones 
-------------------------------

.. py:function:: paddle.fluid.layers.ones(shape,dtype,force_cpu=False)

**ones**

该功能创建一个张量，有具体的维度和dtype，初始值为1。

也将stop_gradient设置为True。

参数：
    - **shape** (tuple|list|None)-输出张量的维
    - **dtype** (np.dtype|core.VarDesc.VarType|str)-输出张量的数据类型

返回：存储在输出中的张量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    data = fluid.layers.ones(shape=[1], dtype='int64')









.. _cn_api_fluid_layers_reverse:

reverse
-------------------------------

.. py:function:: paddle.fluid.layers.reverse(x,axis)

**reverse**

该功能将给定轴上的输入‘x’逆序

参数：
  - **x** (Variable)-预逆序到输入
  - **axis** (int|tuple|list)-其上元素逆序排列的轴。

返回：逆序的张量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

        out = fluid.layers.reverse(x=in, axis=0)
        # or:
        out = fluid.layers.reverse(x=in, axis=[0,1])









.. _cn_api_fluid_layers_sums:

sums
-------------------------------

.. py:function:: paddle.fluid.layers.sums(input,out=None)

该函数对输入进行求和，并返回求和结果作为输出。

参数：
    - **input** (Variable|list)-输入张量，有需要求和的元素
    - **out** (Variable|None)-输出参数。求和结果。默认：None

返回：输入的求和。和参数'out'等同

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    tmp = fluid.layers.zeros(shape=[10], dtype='int32')
    i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
    a0 = layers.array_read(array=tmp, i=i)
    i = layers.increment(x=i)
    a1 = layers.array_read(array=tmp, i=i)
    mean_a0 = layers.mean(a0)
    mean_a1 = layers.mean(a1)
    a_sum = layers.sums(input=[mean_a0, mean_a1])









.. _cn_api_fluid_layers_tensor_array_to_tensor:

tensor_array_to_tensor
-------------------------------

.. py:function:: paddle.fluid.layers.tensor_array_to_tensor(input, axis=1, name=None)

此函数在指定轴上连接LodTensorArray中的元素，并将其作为输出返回。


简单示例如下：

.. code-block:: text

    Given:
    input.data = {[[0.6, 0.1, 0.3],
                   [0.5, 0.3, 0.2]],
                  [[1.3],
                   [1.8]],
                  [[2.3, 2.1],
                   [2.5, 2.4]]}

    axis = 1

    Then:
    output.data = [[0.6, 0.1, 0.3, 1.3, 2.3, 2.1],
                   [0.5, 0.3, 0.2, 1.8, 2.5, 2.4]]
    output_index.data = [3, 1, 2]

参数：
  - **input** (list) - 输入的LodTensorArray
  - **axis** (int) - 整数轴，tensor将会和它连接在一起
  - **name** (str|None) - 该layer的名字，可选。如果设置为none，layer将会被自动命名

返回：
    Variable: 连接的输出变量,输入LodTensorArray沿指定axis连接。

**代码示例：**

.. code-block:: python

   output, output_index = fluid.layers.tensor_array_to_tensor(input=tensor_array)











.. _cn_api_fluid_layers_zeros:

zeros
-------------------------------

.. py:function:: paddle.fluid.layers.zeros(shape,dtype,force_cpu=False)

**zeros**

该功能创建一个张量，含有具体的维度和dtype，初始值为0.

也将stop_gradient设置为True。

参数：
    - **shape** (tuple|list|None)-输出张量的维
    - **dtype** (np.dtype|core.VarDesc.VarType|str)-输出张量的数据类型
    - **force_cpu** (bool,default False)-是否将输出保留在CPU上

返回：存储在输出中的张量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    data = fluid.layers.zeros(shape=[1], dtype='int64')











============
 learning_rate_scheduler 
============


.. _cn_api_fluid_layers_append_LARS:

append_LARS 
-------------------------------

.. py:function:: paddle.fluid.layers.append_LARS(params_grads,learning_rate,weight_decay)

对每一层的学习率运用LARS(LAYER-WISE ADAPTIVE RATE SCALING)



参数：
    - **learning_rate** -变量学习率。LARS的全局学习率。
    - **weight_decay** -Python float类型数

返回： 衰减的学习率

**代码示例** :

.. code-block:: python

        learning_rate*=local_gw_ratio * sqrt(sumsq(param))
            / (sqrt(sumsq(gradient))+ weight_decay * sqrt(sumsq(param)))







.. _cn_api_fluid_layers_exponential_decay:

exponential_decay 
-------------------------------

.. py:function:: paddle.fluid.layers exponential_decay(learning_rate,decay_steps,decay_rate,staircase=False)

在学习率上运用指数衰减。
训练模型时，在训练过程中通常推荐降低学习率。每次 ``decay_steps`` 步骤中用 ``decay_rate`` 衰减学习率。

.. code-block:: text

    if staircase == True:
        decayed_learning_rate = learning_rate * decay_rate ^ floor(global_step / decay_steps)
    else:
        decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)    

参数：
    - **learning_rate** (Variable|float)-初始学习率
    - **decay_steps** (int)-见以上衰减运算
    - **decay_rate** (float)-衰减率。见以上衰减运算
    - **staircase** (Boolean)-若为True,按离散区间衰减学习率。默认：False

返回：衰减的学习率

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    base_lr = 0.1
    sgd_optimizer = fluid.optimizer.SGD(
        learning_rate=fluid.layers.exponential_decay(
            learning_rate=base_lr,
            decay_steps=10000,
            decay_rate=0.5,
            staircase=True))
    sgd_optimizer.minimize(avg_cost)









.. _cn_api_fluid_layers_inverse_time_decay:

inverse_time_decay
-------------------------------

.. py:function:: paddle.fluid.layers.inverse_time_decay(learning_rate, decay_steps, decay_rate, staircase=False)

在初始学习率上运用逆时衰减。

训练模型时，在训练过程中通常推荐降低学习率。通过执行该函数，将对初始学习率运用逆向衰减函数。

.. code-block:: python

    if staircase == True:
         decayed_learning_rate = learning_rate / (1 + decay_rate * floor(global_step / decay_step))
     else:
         decayed_learning_rate = learning_rate / (1 + decay_rate * global_step / decay_step)

参数：
    - **learning_rate** (Variable|float)-初始学习率
    - **decay_steps** (int)-见以上衰减运算
    - **decay_rate** (float)-衰减率。见以上衰减运算
    - **staircase** (Boolean)-若为True，按间隔区间衰减学习率。默认：False

    返回：衰减的学习率

    返回类型：变量（Variable）

**示例代码：**

.. code-block:: python

        base_lr = 0.1
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.layers.inverse_time_decay(
                learning_rate=base_lr,
                decay_steps=10000,
                decay_rate=0.5,
                staircase=True))
        sgd_optimizer.minimize(avg_cost)









.. _cn_api_fluid_layers_natural_exp_decay:

natural_exp_decay
-------------------------------

.. py:function:: paddle.fluid.layers.natural_exp_decay(learning_rate, decay_steps, decay_rate, staircase=False)

将自然指数衰减运用到初始学习率上。

.. code-block:: python

    if not staircase:
        decayed_learning_rate = learning_rate * exp(- decay_rate * (global_step / decay_steps))
    else:
        decayed_learning_rate = learning_rate * exp(- decay_rate * (global_step / decay_steps))

参数：
    - **learning_rate** - 标量float32值或变量。是训练过程中的初始学习率。
    - **decay_steps** - Python int32数
    - **decay_rate** - Python float数
    - **staircase** - Boolean.若设为true，每个decay_steps衰减学习率

返回：衰减的学习率









.. _cn_api_fluid_layers_noam_decay:

noam_decay
-------------------------------

.. py:function:: paddle.fluid.layers noam_decay(d_model,warmup_steps)

Noam衰减方法。noam衰减的numpy实现如下。

.. code-block:: python

    import numpy as np
    lr_value = np.power(d_model, -0.5) * np.min([
                           np.power(current_steps, -0.5),
                           np.power(warmup_steps, -1.5) * current_steps])

请参照 attention is all you need。

参数：
    - **d_model** (Variable)-模型的输入和输出维度
    - **warmup_steps** (Variable)-超参数

返回：衰减的学习率









.. _cn_api_fluid_layers_piecewise_decay:

piecewise_decay
-------------------------------

.. py:function:: paddle.fluid.layers.piecewise_decay(boundaries,values)

对初始学习率进行分段衰减。

该算法可用如下代码描述。

.. code-block:: text

    boundaries = [10000, 20000]
    values = [1.0, 0.5, 0.1]
    if step < 10000:
        learning_rate = 1.0
    elif 10000 <= step < 20000:
        learning_rate = 0.5
    else:
        learning_rate = 0.1

参数：
    - **boundaries** -一列代表步数的数字
    - **values** -一列学习率的值，从不同的步边界中挑选

返回：衰减的学习率









.. _cn_api_fluid_layers_polynomial_decay:

polynomial_decay 
-------------------------------

.. py:function:: paddle.fluid.layers.polynomial_decay(learning_rate,decay_steps,end_learning_rate=0.0001,power=1.0,cycle=False)

对初始学习率使用多项式衰减

.. code-block:: text

    if cycle:
        decay_steps = decay_steps * ceil(global_step / decay_steps)
    else:
        global_step = min(global_step, decay_steps)
        decayed_learning_rate = (learning_rate - end_learning_rate) *
            (1 - global_step / decay_steps) ^ power + end_learning_rate

参数：
    - **learning_rate** (Variable|float32)-标量float32值或变量。是训练过程中的初始学习率。
    - **decay_steps** (int32)-Python int32数
    - **end_learning_rate** (float)-Python float数
    - **power** (float)-Python float数
    - **cycle** (bool)-若设为true，每decay_steps衰减学习率

返回：衰减的学习率

返回类型：变量（Variable）











============
 detection 
============


.. _cn_api_fluid_layers_anchor_generator:

anchor_generator
-------------------------------

.. py:function:: paddle.fluid.layers.anchor_generator(input, anchor_sizes=None, aspect_ratios=None, variance=[0.1, 0.1, 0.2, 0.2], stride=None, offset=0.5, name=None)

**Anchor generator operator**

为快速RCNN算法生成锚，输入的每一位产生N个锚，N=size(anchor_sizes)*size(aspect_ratios)。生成锚的顺序首先是aspect_ratios循环，然后是anchor_sizes循环。

参数：
    - **input** (Variable) - 输入特征图，格式为NCHW
    - **anchor_sizes** (list|tuple|float) - 生成锚的锚大小
    - **in absolute pixels** 等[64.,128.,256.,512.](给定)-实例，锚大小为64意味该锚的面积等于64*2
    - **aspect_ratios** (list|tuple|float) - 生成锚的高宽比，例如[0.5,1.0,2.0]
    - **variance** (list|tuple) - 变量，在框回归delta中使用。默认：[0.1,0.1,0.2,0.2]
    - **stride** (list|tuple) - 锚在宽度和高度方向上的步长，比如[16.0,16.0]
    - **offset** (float) - 先验框的中心位移。默认：0.5
    - **name** (str) - 先验框操作符名称。默认：None

::


    输出anchor，布局[H,W,num_anchors,4]
        H是输入的高度，W是输入的宽度，num_priors是输入每位的框数
        每个anchor格式（非正式格式）为(xmin,ymin,xmax,ymax)
    
::


    变量(Variable):锚的扩展变量
        布局为[H,W,num_priors,4]。H是输入的高度，W是输入的宽度，num_priors是输入每位的框数
	每个变量的格式为(xcenter,ycenter)。

返回类型：anchor（Variable)

**代码示例**：

.. code-block:: python

    anchor, var = anchor_generator(
    input=conv1,
    anchor_sizes=[64, 128, 256, 512],
    aspect_ratios=[0.5, 1.0, 2.0],
    variance=[0.1, 0.1, 0.2, 0.2],
    stride=[16.0, 16.0],
    offset=0.5)









.. _cn_api_fluid_layers_bipartite_match:
        
bipartite_match
-------------------------------

.. py:function:: paddle.fluid.layers.bipartite_match(dist_matrix, match_type=None, dist_threshold=None, name=None)

该算子实现了贪心二分匹配算法，该算法用于根据输入距离矩阵获得与最大距离的匹配。对于输入二维矩阵，二分匹配算法可以找到每一行的匹配列（匹配意味着最大距离），也可以找到每列的匹配行。此运算符仅计算列到行的匹配索引。对于每个实例，匹配索引的数量是输入距离矩阵的列号。

它有两个输出，匹配的索引和距离。简单的描述是该算法将最佳（最大距离）行实体与列实体匹配，并且匹配的索引在ColToRowMatchIndices的每一行中不重复。如果列实体与任何行实体不匹配，则ColToRowMatchIndices设置为-1。

注意：输入距离矩阵可以是LoDTensor（带有LoD）或Tensor。如果LoDTensor带有LoD，则ColToRowMatchIndices的高度是批量大小。如果是Tensor，则ColToRowMatchIndices的高度为1。

注意：此API是一个非常低级别的API。它由 ``ssd_loss`` 层使用。请考虑使用 ``ssd_loss`` 。

参数：
                - **dist_matrix** （变量）- 该输入是具有形状[K，M]的2-D LoDTensor。它是由每行和每列来表示实体之间的成对距离矩阵。例如，假设一个实体是具有形状[K]的A，另一个实体是具有形状[M]的B. dist_matrix [i] [j]是A[i]和B[j]之间的距离。距离越大，匹配越好。

                注意：此张量可以包含LoD信息以表示一批输入。该批次的一个实例可以包含不同数量的实体。

                - **match_type** （string | None）- 匹配方法的类型，应为'bipartite'或'per_prediction'。[默认'二分']。
                - **dist_threshold** （float | None）- 如果match_type为'per_prediction'，则此阈值用于根据最大距离确定额外匹配的bbox，默认值为0.5。

返回：        返回一个包含两个元素的元组。第一个是匹配的索引（matched_indices），第二个是匹配的距离（matched_distance）。

         **matched_indices** 是一个2-D Tensor，int类型的形状为[N，M]。 N是批量大小。如果match_indices[i][j]为-1，则表示B[j]与第i个实例中的任何实体都不匹配。否则，这意味着在第i个实例中B[j]与行match_indices[i][j]匹配。第i个实例的行号保存在match_indices[i][j]中。

         **matched_distance** 是一个2-D Tensor，浮点型的形状为[N，M]。 N是批量大小。如果match_indices[i][j]为-1，则match_distance[i][j]也为-1.0。否则，假设match_distance[i][j]=d，并且每个实例的行偏移称为LoD。然后match_distance[i][j]=dist_matrix[d]+ LoD[i]][j]。

返回类型：        元组(tuple)

**代码示例**

..  code-block:: python

         x = fluid.layers.data(name='x', shape=[4], dtype='float32')
         y = fluid.layers.data(name='y', shape=[4], dtype='float32')
         iou = fluid.layers.iou_similarity(x=x, y=y)
         matched_indices, matched_dist = fluid.layers.bipartite_match(iou)














.. _cn_api_fluid_layers_box_coder:

box_coder
-------------------------------

.. py:function:: paddle.fluid.layers.box_coder(prior_box, prior_box_var, target_box, code_type='encode_center_size', box_normalized=True, name=None)

Bounding Box Coder

编码/解码带有先验框信息的目标边界框

编码规则描述如下：

.. math::

    ox &= (tx - px)/pw/pxv

    oy &= (ty - py)/ph/pyv

    ow &= log(abs(tw/pw))/pwv

    oh &= log(abs(th/ph))/phv

解码规则描述如下：

.. math::

    ox &= (pw * pxv * tx * + px ) - tw/2

    oy &= (ph * pyv * ty * + py ) - th/2

    ow &= exp(pwv * tw ) * pw + tw/2

    oh &= exp(phv * th ) * ph + th/2

其中tx，ty，tw，th分别表示目标框的中心坐标、宽度和高度。同样地，px，py，pw，ph表示先验框地中心坐标、宽度和高度。pxv，pyv，pwv，phv表示先验框变量，ox，oy，ow，oh表示编码/解码坐标、宽度和高度。

参数：
    - **prior_box** (Variable) - 张量，默认float类型的张量。先验框是二维张量，维度为[M,4]，存储M个框，每个框代表[xmin，ymin，xmax，ymax]，[xmin，ymin]是先验框的左顶点坐标，如果输入数图像特征图，则接近坐标原点。[xmax,ymax]是先验框的右底点坐标
    - **prior_box_var** (Variable) - 张量，默认float类型的张量。先验框是二维张量，维度为[M,4]，存储M组变量。PriorBoxVar默认将每个元素置为1
    - **target_box** (Variable) - LoDTensor或者Tensor，当code_type为‘encode_center_size’，输入可以是二维LoDTensor，维度为[N,4]。当code_type为‘decode_center_size’输入可以为三维张量，维度为[N,M,4]。每个框代表[xmin,ymin,xmax,ymax]，[xmin,ymin]是先验框的左顶点坐标，如果输入数图像特征图，则接近坐标原点。[xmax,ymax]是先验框的右底点坐标。该张量包含LoD信息，代表一批输入。批的一个实例可以包含不同的实体数。
    - **code_type** (string，默认encode_center_size) - 编码类型用目标框
    - **box_normalized** (boolean，默认true) - 是否将先验框作为正则框

返回：（LoDTensor 或者 Tensor）

       - ``code_type`` 为 ``‘encode_center_size’`` 时，形为[N,M,4]的输出张量代表N目标框的结果，目标框用M先验框和变量编码。
       - ``code_type`` 为 ``‘decode_center_size’`` 时，N代表batch大小，M代表解码框数

返回类型：output_box（Variable）











.. _cn_api_fluid_layers_density_prior_box:

density_prior_box
-------------------------------

.. py:function:: paddle.fluid.layers.density_prior_box(input, image, densities=None, fixed_sizes=None, fixed_ratios=None, variance=[0.1, 0.1, 0.2, 0.2], clip=False, steps=[0.0, 0.0], offset=0.5, flatten_to_2d=False, name=None)


**Density Prior Box Operator**

为SSD算法(Single Shot MultiBox Detector)生成density prior box。
每个input的位置产生N个prior box，其中，N通过densities, fixed_sizes and fixed_ratios
的量来决定。在每个input位置附近的box center格点，通过此op生成。格点坐标由densities决定，
density prior box的量由fixed_sizes and fixed_ratios决定。显然地，fixed_sizes
和densities相等。对于densities中的densities_i：

.. math::

  N\_density\_prior\_box =sum(N\_fixed\_ratios * {densities\_i}^2)


参数：
  - **input** (Variable) - 输入变量，格式为NCHW
  - **image** (Variable) - PriorBoxOp的输入图像数据，格式为NCHW
  - **densities** (list|tuple|None) - 被生成的density prior boxes的densities，此属性应该是一个整数列表或数组。默认值为None
  - **fixed_sizes** (list|tuple|None) - 被生成的density prior boxes的固定大小，此属性应该为和 :attr:`densities` 有同样长度的列表或数组。默认值为None
  - **fixed_ratios** (list|tuple|None) - 被生成的density prior boxes的固定长度，如果该属性未被设置，同时 :attr:`densities` 和 :attr:`fix_sizes` 被设置，则 :attr:`aspect_ratios` 被用于生成 density prior boxes
  - **variance** (list|tuple) - 将被用于density prior boxes编码的方差，默认值为:[0.1, 0.1, 0.2, 0.2]
  - **clip(bool)** - 是否clip超出范围的box。默认值：False
  - **step** (list|turple) - Prior boxes在宽度和高度的步长，如果step[0] == 0.0/step[1] == 0.0, input的the density prior boxes的高度/宽度的步长将被自动计算。默认值：Default: [0., 0.]
  - **offset** (float) - Prior boxes中心补偿值，默认为：0.5
  - **flatten_to_2d** (bool) - 是否将output prior boxes和方差 ``flatten`` 至2维形状，第二个dim为4。默认值：False
  - **name(str)** - density prior box op的名字，默认值: None

返回：
  tuple: 有两个变量的数组 (boxes, variances)

  boxes: PriorBox的输出density prior boxes

  	当flatten_to_2d为False时，形式为[H, W, num_priors, 4]

  	当flatten_to_2d为True时，形式为[H * W * num_priors, 4]

  	H是输入的高度，W是输入的宽度

  	num_priors是输入中每个位置的总box count

  variances:  PriorBox的expanded variance

  	当flatten_to_2d为False时，形式为[H, W, num_priors, 4]

  	当flatten_to_2d为True时，形式为[H * W * num_priors, 4]

  	H是输入的高度，W是输入的宽度

  	num_priors是输入中每个位置的总box count

**代码示例**:

.. code-block:: python

    box, var = fluid.layers.density_prior_box(
        input=conv1,
        image=images,
        densities=[4, 2, 1],
        fixed_sizes=[32.0, 64.0, 128.0],
        fixed_ratios=[1.],
        clip=True,
        flatten_to_2d=True)











.. _cn_api_fluid_layers_detection_map:
        
detection_map
-------------------------------

.. py:function:: paddle.fluid.layers.detection_map(detect_res, label, function_num, background_label=0, overlap_threshold=0.3, evaluate_difficult=True, has_state=None, input_states=None, out_states=None, ap_version='integral')

检测mAP评估运算符。一般步骤如下：首先，根据检测输入和标签计算TP（true positive）和FP（false positive），然后计算mAP评估值。支持'11 point'和积分mAP算法。请从以下文章中获取更多信息：

        https://sanchom.wordpress.com/tag/average-precision/
        
        https://arxiv.org/abs/1512.02325

参数：
        - **detect_res** （LoDTensor）- 用具有形状[M，6]的2-D LoDTensor来表示检测。每行有6个值：[label，confidence，xmin，ymin，xmax，ymax]，M是此小批量中检测结果的总数。对于每个实例，第一维中的偏移称为LoD，偏移量为N+1，如果LoD[i+1]-LoD[i]== 0，则表示没有检测到数据。
        - **label** （LoDTensor）- 2-D LoDTensor用来带有标签的真实数据。每行有6个值：[label，xmin，ymin，xmax，ymax，is_difficult]或5个值：[label，xmin，ymin，xmax，ymax]，其中N是此小批量中真实数据的总数。对于每个实例，第一维中的偏移称为LoD，偏移量为N + 1，如果LoD [i + 1] - LoD [i] == 0，则表示没有真实数据。
        - **class_num** （int）- 类的数目。
        - **background_label** （int，defalut：0）- background标签的索引，background标签将被忽略。如果设置为-1，则将考虑所有类别。
        - **overlap_threshold** （float）- 检测输出和真实数据下限的重叠阈值。
        - **evaluate_difficult** （bool，默认为true）- 通过切换来控制是否对difficult-data进行评估。
        - **has_state** （Tensor <int>）- 是shape[1]的张量，0表示忽略输入状态，包括PosCount，TruePos，FalsePos。
        - **input_states** - 如果不是None，它包含3个元素：

            1、pos_count（Tensor）是一个shape为[Ncls，1]的张量，存储每类的输入正例的数量，Ncls是输入分类的数量。此输入用于在执行多个小批量累积计算时传递最初小批量生成的AccumPosCount。当输入（PosCount）为空时，不执行累积计算，仅计算当前小批量的结果。
        
            2、true_pos（LoDTensor）是一个shape为[Ntp，2]的2-D LoDTensor，存储每个类输入的正实例。此输入用于在执行多个小批量累积计算时传递最初小批量生成的AccumPosCount。
        
            3、false_pos（LoDTensor）是一个shape为[Nfp，2]的2-D LoDTensor，存储每个类输入的负实例。此输入用于在执行多个小批量累积计算时传递最初小批量生成的AccumPosCount。
        
        - **out_states** - 如果不是None，它包含3个元素：

            1、accum_pos_count（Tensor）是一个shape为[Ncls，1]的Tensor，存储每个类的实例数。它结合了输入（PosCount）和从输入中的（Detection）和（label）计算的正例数。 
        
            2、accum_true_pos（LoDTensor）是一个shape为[Ntp'，2]的LoDTensor，存储每个类的正实例。它结合了输入（TruePos）和从输入中（Detection）和（label）计算的正实例数。 。 
        
            3、accum_false_pos（LoDTensor）是一个shape为[Nfp'，2]的LoDTensor，存储每个类的负实例。它结合了输入（FalsePos）和从输入中（Detection）和（label）计算的负实例数。
        
        - **ap_version** （string，默认'integral'）- AP算法类型，'integral'或'11 point'。

返回：        具有形状[1]的（Tensor），存储mAP的检测评估结果。

**代码示例**

..  code-block:: python

        detect_res = fluid.layers.data(
            name='detect_res',
            shape=[10, 6],
            append_batch_size=False,
            dtype='float32')
        label = fluid.layers.data(
            name='label',
            shape=[10, 6],
            append_batch_size=False,
            dtype='float32')
        map_out = fluid.layers.detection_map(detect_res, label, 21)











.. _cn_api_fluid_layers_detection_output:

detection_output
-------------------------------

.. py:function:: paddle.fluid.layers.detection_output(loc, scores, prior_box, prior_box_var, background_label=0, nms_threshold=0.3, nms_top_k=400, keep_top_k=200, score_threshold=0.01, nms_eta=1.0)

Detection Output Layer for Single Shot Multibox Detector(SSD)

该操作符用于获得检测结果，执行步骤如下：

    1.根据先验框解码输入边界框（bounding box）预测

    2.通过运用多类非最大压缩(NMS)获得最终检测结果

请注意，该操作符不将最终输出边界框剪切至图像窗口。

参数：
    - **loc** (Variable) - 一个三维张量（Tensor），维度为[N,M,4]，代表M个bounding bboxes的预测位置。N是批尺寸，每个边界框（boungding box）有四个坐标值，布局为[xmin,ymin,xmax,ymax]
    - **scores** (Variable) - 一个三维张量（Tensor），维度为[N,M,C]，代表预测置信预测。N是批尺寸，C是类别数，M是边界框数。对每类一共M个分数，对应M个边界框
    - **prior_box** (Variable) - 一个二维张量（Tensor),维度为[M,4]，存储M个框，每个框代表[xmin,ymin,xmax,ymax]，[xmin,ymin]是anchor box的左上坐标，如果输入是图像特征图，靠近坐标系统的原点。[xmax,ymax]是anchor box的右下坐标
    - **prior_box_var** (Variable) - 一个二维张量（Tensor），维度为[M,4]，存有M变量群
    - **background_label** (float) - 背景标签索引，背景标签将会忽略。若设为-1，将考虑所有类别
    - **nms_threshold** (int) - 用于NMS的临界值（threshold）
    - **nms_top_k** (int) - 基于score_threshold过滤检测后，根据置信数维持的最大检测数
    - **keep_top_k** (int) - NMS步后，每一图像要维持的总bbox数
    - **score_threshold** (float) - 临界函数（Threshold），用来过滤带有低置信数的边界框（bounding box）。若未提供，则考虑所有框
    - **nms_eta** (float) - 适应NMS的参数

返回：
	检测输出一个LoDTensor，维度为[No,6]。每行有6个值：[label,confidence,xmin,ymin,xmax,ymax]。No是该mini-batch的总检测数。对每个实例，第一维偏移称为LoD，偏移数为N+1，N是批尺寸。第i个图像有LoD[i+1]-LoD[i]检测结果。如果为0，第i个图像无检测结果。如果所有图像都没有检测结果，LoD所有元素都为0，并且输出张量只包含一个值-1。

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    pb = layers.data(name='prior_box', shape=[10, 4],
             append_batch_size=False, dtype='float32')
    pbv = layers.data(name='prior_box_var', shape=[10, 4],
              append_batch_size=False, dtype='float32')
    loc = layers.data(name='target_box', shape=[2, 21, 4],
              append_batch_size=False, dtype='float32')
    scores = layers.data(name='scores', shape=[2, 21, 10],
              append_batch_size=False, dtype='float32')
    nmsed_outs = fluid.layers.detection_output(scores=scores,
                           loc=loc,
                           prior_box=pb,
                           prior_box_var=pbv)











.. _cn_api_fluid_layers_generate_proposal_labels:

generate_proposal_labels
-------------------------------

.. py:function:: paddle.fluid.layers.generate_proposal_labels(rpn_rois, gt_functiones, is_crowd, gt_boxes, im_info, batch_size_per_im=256, fg_fraction=0.25, fg_thresh=0.25, bg_thresh_hi=0.5, bg_thresh_lo=0.0, bbox_reg_weights=[0.1, 0.1, 0.2, 0.2], function_nums=None, use_random=True)

**该函数可以应用于 Faster-RCNN 网络，生成建议标签。**

该函数可以根据 ``GenerateProposals`` 的输出结果，即bounding boxes（区域框），groundtruth（正确标记数据）来对foreground boxes和background boxes进行采样，并计算loss值。

RpnRois 是RPN的输出box， 并由 ``GenerateProposals`` 来进一步处理, 这些box将与groundtruth boxes合并， 并根据 ``batch_size_per_im`` 和 ``fg_fraction`` 进行采样。 

如果一个实例具有大于 ``fg_thresh`` (前景重叠阀值)的正确标记重叠，那么它会被认定为一个前景样本。
如果一个实例具有的正确标记重叠大于 ``bg_thresh_lo`` 且小于 ``bg_thresh_hi`` (详见参数说明)，那么它将被认定为一个背景样本。
在所有前景、背景框（即Rois regions of interest 直译：有意义的区域）被选择后，我们接着采用随机采样的方法来确保前景框数量不多于 batch_size_per_im * fg_fraction 。

对Rois中的每个box, 我们给它分配类标签和回归目标(box label)。最后 ``bboxInsideWeights`` 和 ``BboxOutsideWeights`` 用来指明是否它将影响训练loss值。

参数:	
  - **rpn_rois** (Variable) – 形为[N, 4]的二维LoDTensor。 N 为 ``GenerateProposals`` 的输出结果, 其中各元素为 :math:`[x_{min}, y_{min}, x_{max}, y_{max}]` 格式的边界框
  - **gt_classes** (Variable) – 形为[M, 1]的二维LoDTensor。 M 为正确标记数据数目, 其中各元素为正确标记数据的类别标签
  - **is_crowd** (Variable) – 形为[M, 1]的二维LoDTensor。M 为正确标记数据数目, 其中各元素为一个标志位，表明一个正确标记数据是不是crowd
  - **gt_boxes** (Variable) – 形为[M, 4]的二维LoDTensor。M 为正确标记数据数目, 其中各元素为 :math:`[x_{min}, y_{min}, x_{max}, y_{max}]` 格式的边界框
  - **im_info** (Variable) – 形为[B, 3]的二维LoDTensor。B 为输入图片的数目, 各元素由 im_height, im_width, im_scale 组成.
  - **batch_size_per_im** (int) – 每张图片的Rois batch数目
  - **fg_fraction** (float) – Foreground前景在 ``batch_size_per_im`` 中所占比例
  - **fg_thresh** (float) – 前景重叠阀值，用于选择foreground前景样本
  - **bg_thresh_hi** (float) – 背景重叠阀值的上界，用于筛选背景样本
  - **bg_thresh_lo** (float) – 背景重叠阀值的下界，用于筛选背景样本O
  - **bbox_reg_weights** (list|tuple) – Box 回归权重
  - **class_nums** (int) – 种类数目
  - **use_random** (bool) – 是否使用随机采样来选择foreground（前景）和background（背景） boxes（框）














.. _cn_api_fluid_layers_generate_proposals:

generate_proposals
-------------------------------

.. py:function:: paddle.fluid.layers.generate_proposals(scores, bbox_deltas, im_info, anchors, variances, pre_nms_top_n=6000, post_nms_top_n=1000, nms_thresh=0.5, min_size=0.1, eta=1.0, name=None) 

生成proposal的Faster-RCNN

该操作根据每个框为foreground（前景）对象的概率，并且通过锚（anchors）来计算这些框，进而提出RoI。Bbox_deltais和一个objects的分数作为是RPN的输出。最终 ``proposals`` 可用于训练检测网络。

为了生成 ``proposals`` ，此操作执行以下步骤：

        1、转置和调整bbox_deltas的分数和大小为（H * W * A，1）和（H * W * A，4）。
        
        2、计算方框位置作为 ``proposals`` 候选框。
        
        3、剪辑框图像。
        
        4、删除小面积的预测框。
        
        5、应用NMS以获得最终 ``proposals`` 作为输出。
        
参数：
        - **scores** (Variable)- 是一个shape为[N，A，H，W]的4-D张量，表示每个框成为object的概率。N是批量大小，A是anchor数，H和W是feature map的高度和宽度。
        - **bbox_deltas** （Variable）- 是一个shape为[N，4 * A，H，W]的4-D张量，表示预测框位置和anchor位置之间的差异。
        - **im_info** （Variable）- 是一个shape为[N，3]的2-D张量，表示N个批次原始图像的信息。信息包含原始图像大小和 ``feature map`` 的大小之间高度，宽度和比例。
        - **anchors** （Variable）- 是一个shape为[H，W，A，4]的4-D Tensor。H和W是 ``feature map`` 的高度和宽度，
        - **num_anchors** - 是每个位置的框的数量。每个anchor都是以非标准化格式（xmin，ymin，xmax，ymax）定义的。
        - **variances** （Variable）- anchor的方差，shape为[H，W，num_priors，4]。每个方差都是（xcenter，ycenter，w，h）这样的格式。
        - **pre_nms_top_n** （float）- 每个图在NMS之前要保留的总框数。默认为6000。 
        - **post_nms_top_n** （float）- 每个图在NMS后要保留的总框数。默认为1000。 
        - **nms_thresh** （float）- NMS中的阈值，默认为0.5。 
        - **min_size** （float）- 删除高度或宽度小于min_size的预测框。默认为0.1。
        - **eta** （float）- 在自适应NMS中应用，如果自适应阈值> 0.5，则在每次迭代中使用adaptive_threshold = adaptive_treshold * eta。











.. _cn_api_fluid_layers_iou_similarity:

iou_similarity
-------------------------------

.. py:function:: paddle.fluid.layers.iou_similarity(x, y, name=None)

**IOU Similarity Operator**

计算两个框列表的intersection-over-union(IOU)。框列表‘X’应为LoDTensor，‘Y’是普通张量，X成批输入的所有实例共享‘Y’中的框。给定框A和框B，IOU的运算如下：

.. math::
    IOU(A, B) = \frac{area(A\cap B)}{area(A)+area(B)-area(A\cap B)}

参数：
    - **x** (Variable,默认LoDTensor,float类型) - 框列表X是二维LoDTensor，shape为[N,4],存有N个框，每个框代表[xmin,ymin,xmax,ymax],X的shape为[N,4]。如果输入是图像特征图,[xmin,ymin]市框的左上角坐标，接近坐标轴的原点。[xmax,ymax]是框的右下角坐标。张量可以包含代表一批输入的LoD信息。该批的一个实例能容纳不同的项数
    - **y** (Variable,张量，默认float类型的张量) - 框列表Y存有M个框，每个框代表[xmin,ymin,xmax,ymax],X的shape为[N,4]。如果输入是图像特征图,[xmin,ymin]市框的左上角坐标，接近坐标轴的原点。[xmax,ymax]是框的右下角坐标。张量可以包含代表一批输入的LoD信息。

返回：iou_similarity操作符的输出，shape为[N,M]的张量，代表一对iou分数

返回类型：out(Variable)









.. _cn_api_fluid_layers_multi_box_head:
        
multi_box_head
-------------------------------

.. py:function:: paddle.fluid.layers.multi_box_head(inputs, image, base_size, num_functiones, aspect_ratios, min_ratio=None, max_ratio=None, min_sizes=None, max_sizes=None, steps=None, step_w=None, step_h=None, offset=0.5, variance=[0.1, 0.1, 0.2, 0.2], flip=True, clip=False, kernel_size=1, pad=0, stride=1, name=None, min_max_aspect_ratios_order=False)

生成SSD（Single Shot MultiBox Detector）算法的候选框。有关此算法的详细信息，请参阅SSD论文 `SSD：Single Shot MultiBox Detector <https://arxiv.org/abs/1512.02325>`_ 的2.2节。

参数：
        - **inputs** （list | tuple）- 输入变量列表，所有变量的格式为NCHW。
        - **image** （Variable）- PriorBoxOp的输入图像数据，布局为NCHW。
        - **base_size** （int）- base_size用于根据 ``min_ratio`` 和 ``max_ratio`` 来获取 ``min_size`` 和 ``max_size`` 。
        - **num_classes** （int）- 类的数量。
        - **aspect_ratios** （list | tuple）- 生成候选框的宽高比。 ``input`` 和 ``aspect_ratios`` 的长度必须相等。
        - **min_ratio** （int）- 生成候选框的最小比率。
        - **max_ratio** （int）- 生成候选框的最大比率。
        - **min_sizes** （list | tuple | None）- 如果len（输入）<= 2，则必须设置 ``min_sizes`` ，并且 ``min_sizes`` 的长度应等于输入的长度。默认值：无。
        - **max_sizes** （list | tuple | None）- 如果len（输入）<= 2，则必须设置 ``max_sizes`` ，并且 ``min_sizes`` 的长度应等于输入的长度。默认值：无。
        - **steps** （list | tuple）- 如果step_w和step_h相同，则step_w和step_h可以被steps替换。
        - **step_w** （list | tuple）- 候选框跨越宽度。如果step_w [i] == 0.0，将自动计算输跨越入[i]宽度。默认值：无。
        - **step_h** （list | tuple）- 候选框跨越高度，如果step_h [i] == 0.0，将自动计算跨越输入[i]高度。默认值：无。
        - **offset** （float）- 候选框中心偏移。默认值：0.5
        - **variance** （list | tuple）- 在候选框编码的方差。默认值：[0.1,0.1,0.2,0.2]。
        - **flip** （bool）- 是否翻转宽高比。默认值：false。
        - **clip** （bool）- 是否剪切超出边界的框。默认值：False。
        - **kernel_size** （int）- conv2d的内核大小。默认值：1。
        - **pad** （int | list | tuple）- conv2d的填充。默认值：0。
        - **stride** （int | list | tuple）- conv2d的步长。默认值：1，
        - **name** （str）- 候选框的名称。默认值：无。
        - **min_max_aspect_ratios_order** （bool）- 如果设置为True，则输出候选框的顺序为[min，max，aspect_ratios]，这与Caffe一致。请注意，此顺序会影响卷积层后面的权重顺序，但不会影响最终检测结果。默认值：False。

返回：一个带有四个变量的元组，（mbox_loc，mbox_conf，boxes, variances）:

    - **mbox_loc** ：预测框的输入位置。布局为[N，H * W * Priors，4]。其中 ``Priors`` 是每个输位置的预测框数。

    - **mbox_conf** ：预测框对输入的置信度。布局为[N，H * W * Priors，C]。其中 ``Priors`` 是每个输入位置的预测框数，C是类的数量。

    - **boxes** ： ``PriorBox`` 的输出候选框。布局是[num_priors，4]。 ``num_priors`` 是每个输入位置的总盒数。

    - **variances** ： ``PriorBox`` 的方差。布局是[num_priors，4]。 ``num_priors`` 是每个输入位置的总窗口数。

返回类型：元组（tuple）
        
**代码示例**

..  code-block:: python

        mbox_locs, mbox_confs, box, var = fluid.layers.multi_box_head(
          inputs=[conv1, conv2, conv3, conv4, conv5, conv5],
          image=images,
          num_classes=21,
          min_ratio=20,
          max_ratio=90,
          aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2.], [2.]],
          base_size=300,
          offset=0.5,
          flip=True,
          clip=True)










.. _cn_api_fluid_layers_polygon_box_transform:

polygon_box_transform
-------------------------------

.. py:function:: paddle.fluid.layers.polygon_box_transform(input, name=None)  

PolygonBoxTransform 算子。

该算子用于将偏移坐标转变为真正的坐标。

输入是检测网络的最终几何输出。我们使用 2*n 个数来表示从 polygon_box 中的 n 个顶点(vertice)到像素位置的偏移。由于每个距离偏移包含两个数字 :math:`(x_i, y_i)` ，所以何输出包含 2*n 个通道。

参数：
    - **input** （Variable） - shape 为[batch_size，geometry_channels，height，width]的张量

返回：与输入 shpae 相同

返回类型：output（Variable）











.. _cn_api_fluid_layers_prior_box:

prior_box 
-------------------------------
.. py:function:: paddle.fluid.layers.prior_box(input,image,min_sizes=None,aspect_ratios=[1.0],variance=[0.1,0.1,0.2,0.2],flip=False,clip=False,steps=[0.0,0.0],offset=0.5,name=None,min_max_aspect_ratios_order=False)

**Prior Box Operator**

为SSD(Single Shot MultiBox Detector)算法生成先验框。输入的每个位产生N个先验框，N由min_sizes,max_sizes和aspect_ratios的数目决定，先验框的尺寸在(min_size,max_size)之间，该尺寸根据aspect_ratios在序列中生成。

参数：
    - **input** (Variable)-输入变量，格式为NCHW
    - **image** (Variable)-PriorBoxOp的输入图像数据，布局为NCHW
    - **min_sizes** (list|tuple|float值)-生成的先验框的最小尺寸
    - **max_sizes** (list|tuple|None)-生成的先验框的最大尺寸。默认：None
    - **aspect_ratios** (list|tuple|float值)-生成的先验框的纵横比。默认：[1.]
    - **variance** (list|tuple)-先验框中的变量，会被解码。默认：[0.1,0.1,0.2,0.2]
    - **flip** (bool)-是否忽略纵横比。默认：False。
    - **clip** (bool)-是否修建溢界框。默认：False。
    - **step** (list|tuple)-先验框在width和height上的步长。如果step[0] == 0.0/step[1] == 0.0，则自动计算先验框在宽度和高度上的步长。默认：[0.,0.]
    - **offset** (float)-先验框中心位移。默认：0.5
    - **name** (str)-先验框操作符名称。默认：None
    - **min_max_aspect_ratios_order** (bool)-若设为True,先验框的输出以[min,max,aspect_ratios]的顺序，和Caffe保持一致。请注意，该顺序会影响后面卷基层的权重顺序，但不影响最后的检测结果。默认：False。

返回：
    含有两个变量的元组(boxes,variances)
    boxes:PriorBox的输出先验框。布局是[H,W,num_priors,4]。H是输入的高度，W是输入的宽度，num_priors是输入每位的总框数
    variances:PriorBox的扩展变量。布局上[H,W,num_priors,4]。H是输入的高度，W是输入的宽度，num_priors是输入每位的总框数

返回类型：元组

**代码示例**：

.. code-block:: python

    box, var = fluid.layers.prior_box(
        input=conv1,
        image=images,
        min_sizes=[100.],
        flip=True,
        clip=True)











.. _cn_api_fluid_layers_roi_perspective_transform:

roi_perspective_transform
-------------------------------

.. py:function:: paddle.fluid.layers.roi_perspective_transform(input, rois, transformed_height, transformed_width, spatial_scale=1.0)

**ROI perspective transform操作符**

参数：
    - **input** (Variable) - ROI Perspective TransformOp的输入。输入张量的形式为NCHW。N是批尺寸，C是输入通道数，H是特征高度，W是特征宽度
    - **rois** (Variable) - 用来处理的ROIs，应该是shape的二维LoDTensor(num_rois,8)。给定[[x1,y1,x2,y2,x3,y3,x4,y4],...],(x1,y1)是左上角坐标，(x2,y2)是右上角坐标，(x3,y3)是右下角坐标，(x4,y4)是左下角坐标
    - **transformed_height** - 输出的宽度
    - **spatial_scale** (float) - 空间尺度因子，用于缩放ROI坐标，默认：1.0。

返回：
   ``ROIPerspectiveTransformOp`` 的输出，带有shape的四维张量(num_rois,channels,transformed_h,transformed_w)

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    out = fluid.layers.roi_perspective_transform(input, rois, 7, 7, 1.0)

















.. _cn_api_fluid_layers_rpn_target_assign:

rpn_target_assign
-------------------------------

.. py:function:: paddle.fluid.layers.rpn_target_assign(bbox_pred, cls_logits, anchor_box, anchor_var, gt_boxes, is_crowd, im_info, rpn_batch_size_per_im=256, rpn_straddle_thresh=0.0, rpn_fg_fraction=0.5, rpn_positive_overlap=0.7, rpn_negative_overlap=0.3, use_random=True)

在Faster-RCNN检测中为区域检测网络（RPN）分配目标层。

对于给定anchors和真实框之间的IoU重叠，该层可以为每个anchors做分类和回归，这些target labels用于训练RPN。classification targets是二进制的类标签（是或不是对象）。根据Faster-RCNN的论文，positive labels有两种anchors：

(i) anchor/anchors与真实框具有最高IoU重叠；

(ii) 具有IoU重叠的anchors高于带有任何真实框（ground-truth box）的rpn_positive_overlap0（0.7）。

请注意，单个真实框（ground-truth box）可以为多个anchors分配正标签。对于所有真实框（ground-truth box），非正向锚是指其IoU比率低于rpn_negative_overlap（0.3）。既不是正也不是负的anchors对训练目标没有价值。回归目标是与positive anchors相关联而编码的图片真实框。

参数：
        - **bbox_pred** （Variable）- 是一个shape为[N，M，4]的3-D Tensor，表示M个边界框的预测位置。N是批量大小，每个边界框有四个坐标值，即[xmin，ymin，xmax，ymax]。
        - **cls_logits** （Variable）- 是一个shape为[N，M，1]的3-D Tensor，表示预测的置信度。N是批量大小，1是frontground和background的sigmoid，M是边界框的数量。
        - **anchor_box** （Variable）- 是一个shape为[M，4]的2-D Tensor，它拥有M个框，每个框可表示为[xmin，ymin，xmax，ymax]，[xmin，ymin]是anchor框的左上部坐标，如果输入是图像特征图，则它们接近坐标系的原点。 [xmax，ymax]是anchor框的右下部坐标。
        - **anchor_var** （Variable）- 是一个shape为[M，4]的2-D Tensor，它拥有anchor的expand方差。
        - **gt_boxes** （Variable）- 真实边界框是一个shape为[Ng，4]的2D LoDTensor，Ng是小批量输入的真实框（bbox）总数。
        - **is_crowd** （Variable）- 1-D LoDTensor，表示（groud-truth）是密集的。
        - **im_info** （Variable）- 是一个形为[N，3]的2-D LoDTensor。N是batch大小，第二维上的3维分别代表高度，宽度和规模(scale)
        - **rpn_batch_size_per_im** （int）- 每个图像中RPN示例总数。
        - **rpn_straddle_thresh** （float）- 通过straddle_thresh像素删除出现在图像外部的RPN anchor。
        - **rpn_fg_fraction** （float）- 为foreground（即class> 0）RoI小批量而标记的目标分数，第0类是background。
        - **rpn_positive_overlap** （float）- 对于一个正例的（anchor, gt box）对，是允许anchors和所有真实框之间最小重叠的。
        - **rpn_negative_overlap** （float）- 对于一个反例的（anchor, gt box）对，是允许anchors和所有真实框之间最大重叠的。

返回:

返回元组 (predicted_scores, predicted_location, target_label, target_bbox, bbox_inside_weight) : 
   - **predicted_scores** 和 **predicted_location** 是RPN的预测结果。 **target_label** 和 **target_bbox** 分别是真实准确数据(ground-truth)。 
   - **predicted_location** 是一个形为[F，4]的2D Tensor， **target_bbox** 的形与 **predicted_location** 相同，F是foreground anchors的数量。 
   - **predicted_scores** 是一个shape为[F + B，1]的2D Tensor， **target_label** 的形与 **predict_scores** 的形相同，B是background anchors的数量，F和B取决于此算子的输入。 
   - **Bbox_inside_weight** 标志着predicted_loction是否为fake_fg（假前景），其形为[F,4]。

返回类型：        元组(tuple)


**代码示例**

..  code-block:: python

        bbox_pred = layers.data(name=’bbox_pred’, shape=[100, 4],
                append_batch_size=False, dtype=’float32’)
        cls_logits = layers.data(name=’cls_logits’, shape=[100, 1],
                append_batch_size=False, dtype=’float32’)
        anchor_box = layers.data(name=’anchor_box’, shape=[20, 4],
                append_batch_size=False, dtype=’float32’)
        gt_boxes = layers.data(name=’gt_boxes’, shape=[10, 4],
                append_batch_size=False, dtype=’float32’)
        loc_pred, score_pred, loc_target, score_target, bbox_inside_weight=
                fluid.layers.rpn_target_assign(bbox_pred=bbox_pred,
                        cls_logits=cls_logits, anchor_box=anchor_box, gt_boxes=gt_boxes)
        
        
        








.. _cn_api_fluid_layers_ssd_loss:
        
ssd_loss
-------------------------------

.. py:function:: paddle.fluid.layers.ssd_loss(location, confidence, gt_box, gt_label, prior_box, prior_box_var=None, background_label=0, overlap_threshold=0.5, neg_pos_ratio=3.0, neg_overlap=0.5, loc_loss_weight=1.0, conf_loss_weight=1.0, match_type='per_prediction', mining_type='max_negative', normalize=True, sample_size=None) 

用于SSD的对象检测算法的多窗口损失层

该层用于计算SSD的损失，给定位置偏移预测，置信度预测，候选框和真实框标签，以及实例挖掘的类型。通过执行以下步骤，返回的损失是本地化损失（或回归损失）和置信度损失（或分类损失）的加权和：

1、通过二分匹配算法查找匹配的边界框。

        1.1、计算真实框与先验框之间的IOU相似度。
        
        1.2、通过二分匹配算法计算匹配的边界框。

2、计算难分样本的置信度

        2.1、根据匹配的索引获取目标标签。
        
        2.2、计算置信度损失。

3、应用实例挖掘来获取负示例索引并更新匹配的索引。

4、分配分类和回归目标

        4.1、根据前面的框编码bbox。
        
        4.2、分配回归目标。
        
        4.3、分配分类目标。
        
5、计算总体客观损失。

        5.1计算置信度损失。
        
        5.1计算本地化损失。
        
        5.3计算总体加权损失。
        
参数：
        - **location** （Variable）- 位置预测是具有形状[N，Np，4]的3D张量，N是批量大小，Np是每个实例的预测总数。 4是坐标值的数量，布局是[xmin，ymin，xmax，ymax]。
        - **confidence**  (Variable) - 置信度预测是具有形状[N，Np，C]，N和Np的3D张量，它们与位置相同，C是类号。
        - **gt_box** （Variable）- 真实框（bbox）是具有形状[Ng，4]的2D LoDTensor，Ng是小批量输入的真实框（bbox）的总数。
        - **gt_label** （Variable）- ground-truth标签是具有形状[Ng，1]的2D LoDTensor。
        - **prior_box** （Variable）- 候选框是具有形状[Np，4]的2D张量。
        - **prior_box_var** （Variable）- 候选框的方差是具有形状[Np，4]的2D张量。
        - **background_label** （int）- background标签的索引，默认为0。
        - **overlap_threshold** （float）- 当找到匹配的盒子，如果 ``match_type`` 为'per_prediction'，请使用 ``overlap_threshold`` 确定额外匹配的bbox。默认为0.5。
        - **neg_pos_ratio** （float）- 负框与正框的比率，仅在 ``mining_type`` 为'max_negative'时使用，3.0由defalut使用。
        - **neg_overlap** （float）- 不匹配预测的负重叠上限。仅当mining_type为'max_negative'时使用，默认为0.5。
        - **loc_loss_weight** （float）- 本地化丢失的权重，默认为1.0。
        - **conf_loss_weight** （float）- 置信度损失的权重，默认为1.0。
        - **match_type** （str）- 训练期间匹配方法的类型应为'bipartite'或'per_prediction'，'per_prediction'由defalut提供。
        - **mining_type** （str）- 硬示例挖掘类型应该是'hard_example'或'max_negative'，现在只支持max_negative。
        - **normalize** （bool）- 是否通过输出位置的总数将SSD丢失标准化，默认为True。
        - **sample_size** （int）- 负框的最大样本大小，仅在 ``mining_type`` 为'hard_example'时使用。

返回：        具有形状[N * Np，1]，N和Np的定位损失和置信度损失的加权和与它们在位置上的相同。

抛出异常：        ``ValueError`` - 如果 ``mining_type`` 是'hard_example'，现在只支持 ``max_negative`` 的挖掘类型。

**代码示例**

..  code-block:: python

         pb = fluid.layers.data(
                           name='prior_box',
                           shape=[10, 4],
                           append_batch_size=False,
                           dtype='float32')
         pbv = fluid.layers.data(
                           name='prior_box_var',
                           shape=[10, 4],
                           append_batch_size=False,
                           dtype='float32')
         loc = fluid.layers.data(name='target_box', shape=[10, 4], dtype='float32')
         scores = fluid.layers.data(name='scores', shape=[10, 21], dtype='float32')
         gt_box = fluid.layers.data(
                 name='gt_box', shape=[4], lod_level=1, dtype='float32')
         gt_label = fluid.layers.data(
                 name='gt_label', shape=[1], lod_level=1, dtype='float32')
         loss = fluid.layers.ssd_loss(loc, scores, gt_box, gt_label, pb, pbv)
        









.. _cn_api_fluid_layers_target_assign:

target_assign
-------------------------------

.. py:function:: paddle.fluid.layers.target_assign(input, matched_indices, negative_indices=None, mismatch_value=None, name=None)

对于给定的目标边界框（bounding box）和标签（label），该操作符对每个预测赋予分类和逻辑回归目标函数以及预测权重。权重具体表示哪个预测无需贡献训练误差。

对于每个实例，根据 ``match_indices`` 和 ``negative_indices`` 赋予输入 ``out`` 和 ``out_weight``。将定输入中每个实例的行偏移称为lod，该操作符执行分类或回归目标函数，执行步骤如下：

1.根据match_indices分配所有输入

.. code-block:: text

    If id = match_indices[i][j] > 0,

        out[i][j][0 : K] = X[lod[i] + id][j % P][0 : K]
        out_weight[i][j] = 1.

    Otherwise,

        out[j][j][0 : K] = {mismatch_value, mismatch_value, ...}
        out_weight[i][j] = 0.

2.如果提供neg_indices，根据neg_indices分配out_weight：

假设neg_indices中每个实例的行偏移称为neg_lod，该实例中第i个实例和neg_indices的每个id如下：

.. code-block:: text

    out[i][id][0 : K] = {mismatch_value, mismatch_value, ...}
    out_weight[i][id] = 1.0

参数：
    - **inputs** (Variable) - 输入为三维LoDTensor，维度为[M,P,K]
    - **matched_indices** (Variable) - 张量（Tensor），整型，输入匹配索引为二维张量（Tensor），类型为整型32位，维度为[N,P]，如果MatchIndices[i][j]为-1，在第i个实例中第j列项不匹配任何行项。
    - **negative_indices** (Variable) - 输入负例索引，可选输入，维度为[Neg,1]，类型为整型32，Neg为负例索引的总数
    - **mismatch_value** (float32) - 为未匹配的位置填充值

返回：返回一个元组（out,out_weight）。out是三维张量，维度为[N,P,K],N和P与neg_indices中的N和P一致，K和输入X中的K一致。如果match_indices[i][j]存在，out_weight是输出权重，维度为[N,P,1]。

返回类型：元组（tuple）

**代码示例**：

.. code-block:: python

    matched_indices, matched_dist = fluid.layers.bipartite_match(iou)
    gt = layers.data(
            name='gt', shape=[1, 1], dtype='int32', lod_level=1)
    trg, trg_weight = layers.target_assign(
                gt, matched_indices, mismatch_value=0)











============
 metric_op 
============


.. _cn_api_fluid_layers_accuracy:

accuracy
-------------------------------

.. py:function:: paddle.fluid.layers.accuracy(input, label, k=1, correct=None, total=None)

accuracy layer。 参考 https://en.wikipedia.org/wiki/Precision_and_recall

使用输入和标签计算准确率。 每个类别中top k 中正确预测的个数。注意：准确率的 dtype 由输入决定。 输入和标签 dtype 可以不同。

参数：
    - **input** (Variable)-该层的输入，即网络的预测。支持 Carry LoD。
    - **label** (Variable)-数据集的标签。
    - **k** (int) - 每个类别的 top k
    - **correct** (Variable)-正确的预测个数。
    - **total** (Variable)-总共的样本数。

返回:	正确率

返回类型:	变量（Variable）

**代码示例**

.. code-block:: python

    data = fluid.layers.data(name="data", shape=[-1, 32, 32], dtype="float32")
    label = fluid.layers.data(name="data", shape=[-1,1], dtype="int32")
    predict = fluid.layers.fc(input=data, size=10)
    acc = fluid.layers.accuracy(input=predict, label=label, k=5)












.. _cn_api_fluid_layers_auc:

auc
-------------------------------

.. py:function:: paddle.fluid.layers.auc(input, label, curve='ROC', num_thresholds=4095, topk=1, slide_steps=1)

**Area Under the Curve(AUC) Layer**

该层根据前向输出和标签计算AUC，在二分类(binary classification)估计中广泛使用。

注：如果输入标注包含一种值，只有0或1两种情况，数据类型则强制转换成布尔值。相关定义可以在这里: https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve 找到

有两种可能的曲线：

1. ROC:受试者工作特征曲线

2. PR:准确率召回率曲线

参数：
    - **input** (Variable) - 浮点二维变量，值的范围为[0,1]。每一行降序排列。输入应为topk的输出。该变量显示了每个标签的概率。
    - **label** (Variable) - 二维整型变量，表示训练数据的标注。批尺寸的高度和宽度始终为1.
    - **curve** (str) - 曲线类型，可以为 ``ROC`` 或 ``PR``，默认 ``ROC``。
    - **num_thresholds** (int) - 将roc曲线离散化时使用的临界值数。默认200
    - **topk** (int) - 只有预测输出的topk数才被用于auc
    - **slide_steps** - 计算批auc时，不仅用当前步也用先前步。slide_steps=1，表示用当前步；slide_steps = 3表示用当前步和前两步；slide_steps = 0，则用所有步

返回：代表当前AUC的scalar

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    #  network为二分类模型, label为ground truth（正确标记）
    prediction = network(image, is_infer=True)
    auc_out=fluid.layers.auc(input=prediction, label=label)











