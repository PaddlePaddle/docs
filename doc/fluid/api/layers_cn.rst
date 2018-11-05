

.. _cn_api_fluid_layers_create_array:

create_array
>>>>>>>>>>>>

paddle.fluid.layers.create_array(dtype)
""""""""""""""""""""""""""""""""""""""""""

创建LoDTensorArray数组。它主要用于实现RNN与array_write, array_read和While。

  参数: dtype(int |float)——lod_tensor_array中元素的数据类型。

  返回: lod_tensor_array， 元素数据类型为dtype。

  返回类型: Variable。


**代码示例**

..  code-block:: python
  
  data = fluid.layers.create_array(dtype='float32')
  
  

.. _cn_api_fluid_layers_DynamicRNN:

DynamicRNN
>>>>>>>>>>>>

class paddle.fluid.layers.DynamicRNN(name=None)
""""""""""""""""""""""""""""""""""""""""""

动态RNN可以处理一批序列数据,每个样本序列的长度可以不同。这个API自动批量处理它们。

必须设置输入lod，请参考lod_tensor

**代码示例**

..  code-block:: python
  
>>> import paddle.fluid as fluid
>>> data = fluid.layers.data(name='sentence', dtype='int64', lod_level=1)
>>> embedding = fluid.layers.embedding(input=data, size=[65535, 32],
>>>                                    is_sparse=True)
>>>
>>> drnn = fluid.layers.DynamicRNN()
>>> with drnn.block():
>>>     word = drnn.step_input(embedding)
>>>     prev = drnn.memory(shape=[200])
>>>     hidden = fluid.layers.fc(input=[word, prev], size=200, act='relu')
>>>     drnn.update_memory(prev, hidden)  # set prev to hidden
>>>     drnn.output(hidden)
>>>
>>> # last is the last time step of rnn. It is the encoding result.
>>> last = fluid.layers.sequence_last_step(drnn())


动态RNN将按照timesteps展开开序列。用户需要在with block中定义如何处理处理每个timestep。

memory用于缓存分段数据。memory的初始值可以是零，也可以是其他变量。

动态RNN可以将多个变量标记为其输出。使用drnn()获得输出序列。

  step_input(x)
  
    将序列标记为动态RNN输入。:参数x:输入序列。:x型:变量
    
    返回:当前的输入序列中的timestep。

  static_input(x)
  
    将变量标记为RNN输入。输入不会分散到timestep中。参数x:输入变量。:x型:Variable

    返回:可以访问的RNN的输入变量,。

  block(*args, **kwds)

    用户在RNN中定义operators的block。有关详细信息，请参阅class docstring 。
    
  memory(init=None, shape=None, value=0.0, need_reorder=False, dtype='float32')

    为动态rnn创建一个memory 变量。
    
    如果init不是None，memory将由这个变量初始化。参数need_reorder用于将memory重新排序作为输入变量。当memory初始化依赖于输入样本时，应该将其设置为true。

**例如**

..  code-block:: python
  
>>> import paddle.fluid as fluid
>>> sentence = fluid.layers.data(
>>>                 name='sentence', dtype='float32', shape=[32])
>>> boot_memory = fluid.layers.data(
>>>                 name='boot', dtype='float32', shape=[10])
>>>
>>> drnn = fluid.layers.DynamicRNN()
>>> with drnn.block():
>>>     word = drnn.step_input(sentence)
>>>     memory = drnn.memory(init=boot_memory, need_reorder=True)
>>>     hidden = fluid.layers.fc(
>>>                 input=[word, memory], size=10, act='tanh')
>>>     drnn.update_memory(ex_mem=memory, new_mem=hidden)
>>>     drnn.output(hidden)
>>> rnn_output = drnn()

  否则，如果已经设置shape value dtype，memory将被value初始化
  
..  code-block:: python
  
>>> import paddle.fluid as fluid
>>> sentence = fluid.layers.data(
>>>                 name='sentence', dtype='float32', shape=[32])
>>>
>>> drnn = fluid.layers.DynamicRNN()
>>> with drnn.block():
>>>     word = drnn.step_input(sentence)
>>>     memory = drnn.memory(shape=[10], dtype='float32', value=0)
>>>     hidden = fluid.layers.fc(
>>>             input=[word, memory], size=10, act='tanh')
>>>     drnn.update_memory(ex_mem=memory, new_mem=hidden)
>>>     drnn.output(hidden)
>>> rnn_output = drnn()


  参数：
    - init (Variable|None) – 初始化的Variable.
    - shape (list|tuple) – memory shape. 注意形状不包含
    - batch_size. –batch的大小
    - value (float) – 初始化的值.
    - need_reorder (bool) –memory 初始化依赖于输入样本时设置为True
    - sample. (input) – 输入
    - dtype (str|numpy.dtype) –初始化memory的数据类型

  返回：memory Variable


  update_memory(ex_mem, new_mem)
  
    将内存从ex_mem更新到new_mem。注意，ex_mem和new_mem的shape和数据类型必须相同。:param ex_mem:memory Variable。:param ex_mem: the memory variable. :type ex_mem: Variable :param new_mem: the plain variable generated in RNN block. :type new_mem: Variable

  返回：None


  output(*outputs)
  
    标记RNN输出变量。
    参数:outputs,输出变量。
    返回:None
 
 
.. _cn_api_fluid_layers_StaticRNN:

StaticRNN
>>>>>>>>>>>>

class paddle.fluid.layers.StaticRNN(name=None)
""""""""""""""""""""""""""""""""""""""""""

用于创建static RNN。RNN将有自己的参数，比如输入、输出、memory、状态和长度。

  memory(init=None, shape=None, batch_ref=None, init_value=0.0, init_batch_dim_idx=0, ref_batch_dim_idx=1)
  参数：
  
    - init - boot memory，如果没有设置，则必须提供一个shape
    - shape - boot memory的形状
    - batch_ref - batch引用
    - init_value - boot memory的初始化值
    - init_batch_dim_idx - init维度中的batch大小的索引
    - ref_batch_dim_idx - batch_ref维度中的batch大小索引



 
