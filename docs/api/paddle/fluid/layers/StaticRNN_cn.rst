.. _cn_api_fluid_layers_StaticRNN:

StaticRNN
-------------------------------


.. py:class:: paddle.fluid.layers.StaticRNN(name=None)




该OP用来处理一批序列数据，其中每个样本序列的长度必须相等。StaticRNN将序列按照时间步长展开，用户需要定义每个时间步中的处理逻辑。

参数
::::::::::::

  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

代码示例
::::::::::::

.. code-block:: python

      import paddle.fluid as fluid
      import paddle.fluid.layers as layers

      vocab_size, hidden_size=10000, 200
      x = layers.data(name="x", shape=[-1, 1, 1], dtype='int64')

      # 创建处理用的word sequence
      x_emb = layers.embedding(
          input=x,
          size=[vocab_size, hidden_size],
          dtype='float32',
          is_sparse=False)
      # 把batch size变换到第1维。
      x_emb = layers.transpose(x_emb, perm=[1, 0, 2])

      rnn = fluid.layers.StaticRNN()
      with rnn.step():
          # 将刚才创建的word sequence标记为输入，每个时间步取一个word处理。
          word = rnn.step_input(x_emb)
          # 创建memory变量作为prev，batch size来自于word变量。
          prev = rnn.memory(shape=[-1, hidden_size], batch_ref = word)
          hidden = fluid.layers.fc(input=[word, prev], size=hidden_size, act='relu')
          # 用处理完的hidden变量更新prev变量。
          rnn.update_memory(prev, hidden)
          # 把每一步处理后的hidden标记为输出序列。
          rnn.step_output(hidden)
      # 获取最终的输出结果
      result = rnn()

方法
::::::::::::
step()
'''''''''

定义在每个时间步执行的操作。step用在with语句中，with语句中定义的OP会被执行sequence_len次(sequence_len是输入序列的长度)。


memory(init=None, shape=None, batch_ref=None, init_value=0.0, init_batch_dim_idx=0, ref_batch_dim_idx=1)
'''''''''
 
为静态RNN创建一个内存变量。
如果init不为None，则用init将初始化memory。如果init为None，则必须设置shape和batch_ref，函数会使用shape和batch_ref创建新的Variable来初始化init。

**参数**

  - **init** (Variable，可选) - 用来初始化memory的Tensor。如果没有设置，则必须提供shape和batch_ref参数。默认值None。
  - **shape** (list|tuple) - 当init为None时用来设置memory的维度，注意不包括batch_size。默认值None。
  - **batch_ref** (Variable，可选) - 当init为None时，memory变量的batch size会设置为该batch_ref变量的ref_batch_dim_idx轴。默认值None。
  - **init_value** (float，可选) - 当init为None时用来设置memory的初始值，默认值0.0。
  - **init_batch_dim_idx** (int，可选) - init变量的batch_size轴，默认值0。
  - **ref_batch_dim_idx** (int，可选) - batch_ref变量的batch_size轴，默认值1。

**返回**
返回创建的memory变量。

**返回类型**
Variable


代码示例一
::::::::::::

.. code-block:: python

      import paddle.fluid as fluid
      import paddle.fluid.layers as layers

      vocab_size, hidden_size=10000, 200
      x = layers.data(name="x", shape=[-1, 1, 1], dtype='int64')

      # 创建处理用的word sequence
      x_emb = layers.embedding(
          input=x,
          size=[vocab_size, hidden_size],
          dtype='float32',
          is_sparse=False)
      # 把batch size变换到第1维。
      x_emb = layers.transpose(x_emb, perm=[1, 0, 2])

      rnn = fluid.layers.StaticRNN()
      with rnn.step():
          # 将刚才创建的word sequence标记为输入，每个时间步取一个word处理。
          word = rnn.step_input(x_emb)
          # 创建memory变量作为prev，batch size来自于word变量。
          prev = rnn.memory(shape=[-1, hidden_size], batch_ref = word)
          hidden = fluid.layers.fc(input=[word, prev], size=hidden_size, act='relu')
          # 用处理完的hidden变量更新prev变量。
          rnn.update_memory(prev, hidden)

代码示例二
::::::::::::

.. code-block:: python

      import paddle.fluid as fluid
      import paddle.fluid.layers as layers

      vocab_size, hidden_size=10000, 200
      x = layers.data(name="x", shape=[-1, 1, 1], dtype='int64')

      # 创建处理用的word sequence
      x_emb = layers.embedding(
          input=x,
          size=[vocab_size, hidden_size],
          dtype='float32',
          is_sparse=False)
      # 把batch size变换到第1维。
      x_emb = layers.transpose(x_emb, perm=[1, 0, 2])
      boot_memory = fluid.layers.data(name='boot', shape=[hidden_size], dtype='float32', lod_level=1)

      rnn = fluid.layers.StaticRNN()
      with rnn.step():
          # 将刚才创建的word sequence标记为输入，每个时间步取一个word处理。
          word = rnn.step_input(x_emb)
          # 用init初始化memory。
          prev = rnn.memory(init=boot_memory)
          hidden = fluid.layers.fc(input=[word, prev], size=hidden_size, act='relu')
          # 用处理完的hidden变量更新prev变量。
          rnn.update_memory(prev, hidden)

step_input(x)
'''''''''

标记StaticRNN的输入序列。

**参数**

  - **x** (Variable) – 输入序列，x的形状应为[seq_len, ...]。

**返回**
输入序列中当前时间步的数据。

**返回类型**
Variable


**代码示例**

.. code-block:: python

      import paddle.fluid as fluid
      import paddle.fluid.layers as layers

      vocab_size, hidden_size=10000, 200
      x = layers.data(name="x", shape=[-1, 1, 1], dtype='int64')

      # 创建处理用的word sequence
      x_emb = layers.embedding(
          input=x,
          size=[vocab_size, hidden_size],
          dtype='float32',
          is_sparse=False)
      # 把batch size变换到第1维。
      x_emb = layers.transpose(x_emb, perm=[1, 0, 2])

      rnn = fluid.layers.StaticRNN()
      with rnn.step():
          # 将刚才创建的word sequence标记为输入，每个时间步取一个word处理。
          word = rnn.step_input(x_emb)
          # 创建memory变量作为prev，batch size来自于word变量。
          prev = rnn.memory(shape=[-1, hidden_size], batch_ref = word)
          hidden = fluid.layers.fc(input=[word, prev], size=hidden_size, act='relu')
          # 用处理完的hidden变量更新prev变量。
          rnn.update_memory(prev, hidden)

step_output(o)
'''''''''

标记StaticRNN输出的序列。

**参数**

  -**o** (Variable) – 输出序列

**返回**
无


**代码示例**

.. code-block:: python

      import paddle.fluid as fluid
      import paddle.fluid.layers as layers

      vocab_size, hidden_size=10000, 200
      x = layers.data(name="x", shape=[-1, 1, 1], dtype='int64')

      # 创建处理用的word sequence
      x_emb = layers.embedding(
          input=x,
          size=[vocab_size, hidden_size],
          dtype='float32',
          is_sparse=False)
      # 把batch size变换到第1维。
      x_emb = layers.transpose(x_emb, perm=[1, 0, 2])

      rnn = fluid.layers.StaticRNN()
      with rnn.step():
          # 将刚才创建的word sequence标记为输入，每个时间步取一个word处理。
          word = rnn.step_input(x_emb)
          # 创建memory变量作为prev，batch size来自于word变量。
          prev = rnn.memory(shape=[-1, hidden_size], batch_ref = word)
          hidden = fluid.layers.fc(input=[word, prev], size=hidden_size, act='relu')
          # 用处理完的hidden变量更新prev变量。
          rnn.update_memory(prev, hidden)
          # 把每一步处理后的hidden标记为输出序列。
          rnn.step_output(hidden)

      result = rnn()

output(*outputs)
'''''''''

标记StaticRNN输出变量。

**参数**

  -**outputs** – 输出Tensor，可同时将多个Variable标记为输出。

**返回**
无


**代码示例**

.. code-block:: python

      import paddle.fluid as fluid
      import paddle.fluid.layers as layers

      vocab_size, hidden_size=10000, 200
      x = layers.data(name="x", shape=[-1, 1, 1], dtype='int64')

      # 创建处理用的word sequence
      x_emb = layers.embedding(
          input=x,
          size=[vocab_size, hidden_size],
          dtype='float32',
          is_sparse=False)
      # 把batch size变换到第1维。
      x_emb = layers.transpose(x_emb, perm=[1, 0, 2])

      rnn = fluid.layers.StaticRNN()
      with rnn.step():
          # 将刚才创建的word sequence标记为输入，每个时间步取一个word处理。
          word = rnn.step_input(x_emb)
          # 创建memory变量作为prev，batch size来自于word变量。
          prev = rnn.memory(shape=[-1, hidden_size], batch_ref = word)
          hidden = fluid.layers.fc(input=[word, prev], size=hidden_size, act='relu')
          # 用处理完的hidden变量更新prev变量。
          rnn.update_memory(prev, hidden)
          # 把每一步的hidden和word标记为输出。
          rnn.output(hidden, word)

      result = rnn()


update_memory(mem, var)
'''''''''


将memory从mem更新为var。

**参数**
    
  - **mem** (Variable) – memory接口定义的变量。
  - **var** (Variable) – RNN块中的变量，用来更新memory。var的维度和数据类型必须与mem一致。

**返回**
无

代码示例参考前述示例。

