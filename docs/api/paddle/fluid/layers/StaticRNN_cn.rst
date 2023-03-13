.. _cn_api_fluid_layers_StaticRNN:

StaticRNN
-------------------------------


.. py:class:: paddle.fluid.layers.StaticRNN(name=None)




该 OP 用来处理一批序列数据，其中每个样本序列的长度必须相等。StaticRNN 将序列按照时间步长展开，用户需要定义每个时间步中的处理逻辑。

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

      # 创建处理用的 word sequence
      x_emb = layers.embedding(
          input=x,
          size=[vocab_size, hidden_size],
          dtype='float32',
          is_sparse=False)
      # 把 batch size 变换到第 1 维。
      x_emb = layers.transpose(x_emb, perm=[1, 0, 2])

      rnn = fluid.layers.StaticRNN()
      with rnn.step():
          # 将刚才创建的 word sequence 标记为输入，每个时间步取一个 word 处理。
          word = rnn.step_input(x_emb)
          # 创建 memory 变量作为 prev，batch size 来自于 word 变量。
          prev = rnn.memory(shape=[-1, hidden_size], batch_ref = word)
          hidden = fluid.layers.fc(input=[word, prev], size=hidden_size, act='relu')
          # 用处理完的 hidden 变量更新 prev 变量。
          rnn.update_memory(prev, hidden)
          # 把每一步处理后的 hidden 标记为输出序列。
          rnn.step_output(hidden)
      # 获取最终的输出结果
      result = rnn()

方法
::::::::::::
step()
'''''''''

定义在每个时间步执行的操作。step 用在 with 语句中，with 语句中定义的 OP 会被执行 sequence_len 次(sequence_len 是输入序列的长度)。


memory(init=None, shape=None, batch_ref=None, init_value=0.0, init_batch_dim_idx=0, ref_batch_dim_idx=1)
'''''''''

为静态 RNN 创建一个内存变量。
如果 init 不为 None，则用 init 将初始化 memory。如果 init 为 None，则必须设置 shape 和 batch_ref，函数会使用 shape 和 batch_ref 创建新的 Variable 来初始化 init。

**参数**

  - **init** (Variable，可选) - 用来初始化 memory 的 Tensor。如果没有设置，则必须提供 shape 和 batch_ref 参数。默认值 None。
  - **shape** (list|tuple) - 当 init 为 None 时用来设置 memory 的维度，注意不包括 batch_size。默认值 None。
  - **batch_ref** (Variable，可选) - 当 init 为 None 时，memory 变量的 batch size 会设置为该 batch_ref 变量的 ref_batch_dim_idx 轴。默认值 None。
  - **init_value** (float，可选) - 当 init 为 None 时用来设置 memory 的初始值，默认值 0.0。
  - **init_batch_dim_idx** (int，可选) - init 变量的 batch_size 轴，默认值 0。
  - **ref_batch_dim_idx** (int，可选) - batch_ref 变量的 batch_size 轴，默认值 1。

**返回**
返回创建的 memory 变量。

**返回类型**
Variable


代码示例一
::::::::::::

.. code-block:: python

      import paddle.fluid as fluid
      import paddle.fluid.layers as layers

      vocab_size, hidden_size=10000, 200
      x = layers.data(name="x", shape=[-1, 1, 1], dtype='int64')

      # 创建处理用的 word sequence
      x_emb = layers.embedding(
          input=x,
          size=[vocab_size, hidden_size],
          dtype='float32',
          is_sparse=False)
      # 把 batch size 变换到第 1 维。
      x_emb = layers.transpose(x_emb, perm=[1, 0, 2])

      rnn = fluid.layers.StaticRNN()
      with rnn.step():
          # 将刚才创建的 word sequence 标记为输入，每个时间步取一个 word 处理。
          word = rnn.step_input(x_emb)
          # 创建 memory 变量作为 prev，batch size 来自于 word 变量。
          prev = rnn.memory(shape=[-1, hidden_size], batch_ref = word)
          hidden = fluid.layers.fc(input=[word, prev], size=hidden_size, act='relu')
          # 用处理完的 hidden 变量更新 prev 变量。
          rnn.update_memory(prev, hidden)

代码示例二
::::::::::::

.. code-block:: python

      import paddle.fluid as fluid
      import paddle.fluid.layers as layers

      vocab_size, hidden_size=10000, 200
      x = layers.data(name="x", shape=[-1, 1, 1], dtype='int64')

      # 创建处理用的 word sequence
      x_emb = layers.embedding(
          input=x,
          size=[vocab_size, hidden_size],
          dtype='float32',
          is_sparse=False)
      # 把 batch size 变换到第 1 维。
      x_emb = layers.transpose(x_emb, perm=[1, 0, 2])
      boot_memory = fluid.layers.data(name='boot', shape=[hidden_size], dtype='float32', lod_level=1)

      rnn = fluid.layers.StaticRNN()
      with rnn.step():
          # 将刚才创建的 word sequence 标记为输入，每个时间步取一个 word 处理。
          word = rnn.step_input(x_emb)
          # 用 init 初始化 memory。
          prev = rnn.memory(init=boot_memory)
          hidden = fluid.layers.fc(input=[word, prev], size=hidden_size, act='relu')
          # 用处理完的 hidden 变量更新 prev 变量。
          rnn.update_memory(prev, hidden)

step_input(x)
'''''''''

标记 StaticRNN 的输入序列。

**参数**

  - **x** (Variable) – 输入序列，x 的形状应为[seq_len, ...]。

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

      # 创建处理用的 word sequence
      x_emb = layers.embedding(
          input=x,
          size=[vocab_size, hidden_size],
          dtype='float32',
          is_sparse=False)
      # 把 batch size 变换到第 1 维。
      x_emb = layers.transpose(x_emb, perm=[1, 0, 2])

      rnn = fluid.layers.StaticRNN()
      with rnn.step():
          # 将刚才创建的 word sequence 标记为输入，每个时间步取一个 word 处理。
          word = rnn.step_input(x_emb)
          # 创建 memory 变量作为 prev，batch size 来自于 word 变量。
          prev = rnn.memory(shape=[-1, hidden_size], batch_ref = word)
          hidden = fluid.layers.fc(input=[word, prev], size=hidden_size, act='relu')
          # 用处理完的 hidden 变量更新 prev 变量。
          rnn.update_memory(prev, hidden)

step_output(o)
'''''''''

标记 StaticRNN 输出的序列。

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

      # 创建处理用的 word sequence
      x_emb = layers.embedding(
          input=x,
          size=[vocab_size, hidden_size],
          dtype='float32',
          is_sparse=False)
      # 把 batch size 变换到第 1 维。
      x_emb = layers.transpose(x_emb, perm=[1, 0, 2])

      rnn = fluid.layers.StaticRNN()
      with rnn.step():
          # 将刚才创建的 word sequence 标记为输入，每个时间步取一个 word 处理。
          word = rnn.step_input(x_emb)
          # 创建 memory 变量作为 prev，batch size 来自于 word 变量。
          prev = rnn.memory(shape=[-1, hidden_size], batch_ref = word)
          hidden = fluid.layers.fc(input=[word, prev], size=hidden_size, act='relu')
          # 用处理完的 hidden 变量更新 prev 变量。
          rnn.update_memory(prev, hidden)
          # 把每一步处理后的 hidden 标记为输出序列。
          rnn.step_output(hidden)

      result = rnn()

output(*outputs)
'''''''''

标记 StaticRNN 输出变量。

**参数**

  -**outputs** – 输出 Tensor，可同时将多个 Variable 标记为输出。

**返回**
无


**代码示例**

.. code-block:: python

      import paddle.fluid as fluid
      import paddle.fluid.layers as layers

      vocab_size, hidden_size=10000, 200
      x = layers.data(name="x", shape=[-1, 1, 1], dtype='int64')

      # 创建处理用的 word sequence
      x_emb = layers.embedding(
          input=x,
          size=[vocab_size, hidden_size],
          dtype='float32',
          is_sparse=False)
      # 把 batch size 变换到第 1 维。
      x_emb = layers.transpose(x_emb, perm=[1, 0, 2])

      rnn = fluid.layers.StaticRNN()
      with rnn.step():
          # 将刚才创建的 word sequence 标记为输入，每个时间步取一个 word 处理。
          word = rnn.step_input(x_emb)
          # 创建 memory 变量作为 prev，batch size 来自于 word 变量。
          prev = rnn.memory(shape=[-1, hidden_size], batch_ref = word)
          hidden = fluid.layers.fc(input=[word, prev], size=hidden_size, act='relu')
          # 用处理完的 hidden 变量更新 prev 变量。
          rnn.update_memory(prev, hidden)
          # 把每一步的 hidden 和 word 标记为输出。
          rnn.output(hidden, word)

      result = rnn()


update_memory(mem, var)
'''''''''


将 memory 从 mem 更新为 var。

**参数**

  - **mem** (Variable) – memory 接口定义的变量。
  - **var** (Variable) – RNN 块中的变量，用来更新 memory。var 的维度和数据类型必须与 mem 一致。

**返回**
无

代码示例参考前述示例。
