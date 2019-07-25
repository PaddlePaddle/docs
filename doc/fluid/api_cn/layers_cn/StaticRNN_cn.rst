.. _cn_api_fluid_layers_StaticRNN:

StaticRNN
-------------------------------

.. py:class:: paddle.fluid.layers.StaticRNN(name=None)

StaticRNN可以处理一批序列数据。每个样本序列的长度必须相等。StaticRNN将拥有自己的参数，如输入、输出和存储器等。请注意，输入的第一个维度表示序列长度，且输入的所有序列长度必须相同。并且输入和输出的每个轴的含义是相同的。

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        import paddle.fluid.layers as layers
        
        vocab_size, hidden_size=10000, 200
        x = layers.data(name="x", shape=[-1, 1, 1], dtype='int64')
        x_emb = layers.embedding(
                input=x,
                size=[vocab_size, hidden_size],
                dtype='float32',
                is_sparse=False)
        x_emb = layers.transpose(x_emb, perm=[1, 0, 2])
      
        rnn = fluid.layers.StaticRNN()
        with rnn.step():
           word = rnn.step_input(x_emb)
           prev = rnn.memory(shape=[-1, hidden_size], batch_ref = word)
           hidden = fluid.layers.fc(input=[word, prev], size=hidden_size, act='relu')
           rnn.update_memory(prev, hidden)  # set prev to hidden
           rnn.step_output(hidden)
        
        result = rnn()

StaticRNN将序列展开为时间步长。用户需要定义如何在with步骤中处理每个时间步长。

内存用作在time step之间缓存数据。内存的初始值可以是填充常量值的变量或指定变量。

StaticRNN可以将多个变量标记为其输出。使用rnn()获取输出序列。


.. py:method:: step()

  用户在该代码块中定义RNN中的operators。


.. py:method:: memory(init=None, shape=None, batch_ref=None, init_value=0.0, init_batch_dim_idx=0, ref_batch_dim_idx=1)
 
  为静态RNN创建一个内存变量。
  如果init不为None，则此变量将初始化内存。 如果init为None，则必须设置shape和batch_ref，并且此函数将初始化init变量。

  参数：
    - **init** (Variable|None) - 初始化过的变量，如果没有设置，则必须提供shape和batch_ref，默认值None
    - **shape** (list|tuple) - boot memory的形状，注意其不包括batch_size，默认值None
    - **batch_ref** (Variable|None) - batch引用变量，默认值None
    - **init_value** (float) - boot memory的初始化值，默认值0.0
    - **init_batch_dim_idx** (int) - init变量的batch_size轴，默认值0
    - **ref_batch_dim_idx** (int) - batch_ref变量的batch_size轴

  返回：内存变量


  **代码示例**：

  .. code-block:: python

        import paddle.fluid as fluid
        import paddle.fluid.layers as layers

        vocab_size, hidden_size=10000, 200
        x = layers.data(name="x", shape=[-1, 1, 1], dtype='int64')
        x_emb = layers.embedding(
            input=x,
            size=[vocab_size, hidden_size],
            dtype='float32',
            is_sparse=False)
        x_emb = layers.transpose(x_emb, perm=[1, 0, 2])

        rnn = fluid.layers.StaticRNN()
        with rnn.step():
            word = rnn.step_input(x_emb)
            prev = rnn.memory(shape=[-1, hidden_size], batch_ref = word)
            hidden = fluid.layers.fc(input=[word, prev], size=hidden_size, act='relu')
            rnn.update_memory(prev, hidden)

.. py:method:: step_input(x)

  标记作为StaticRNN输入的序列。

  参数：
    - **x** (Variable) – 输入序列，x的形状应为[seq_len, ...]。

  返回：输入序列中的当前时间步长。



.. py:method:: step_output(o)

  标记作为StaticRNN输出的序列。

  参数：
    -**o** (Variable) – 输出序列

  返回：None


.. py:method:: output(*outputs)

  标记StaticRNN输出变量。

  参数：
    -**outputs** – 输出变量

  返回：None


.. py:method:: update_memory(mem, var)

  将内存从ex_mem更新为new_mem。请注意，ex_mem和new_mem的形状和数据类型必须相同。

  参数：    
    - **mem** (Variable) – 内存变量
    - **var** (Variable) – RNN块中产生的普通变量

  返回：None










