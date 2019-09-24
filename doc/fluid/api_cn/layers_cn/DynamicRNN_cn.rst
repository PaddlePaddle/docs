.. _cn_api_fluid_layers_DynamicRNN:

DynamicRNN
-------------------------------

.. py:class:: paddle.fluid.layers.DynamicRNN(name=None)

**该OP仅支持LoDTensor，即要求输入数据的LoD信息不为空**。输入数据不是LoDTensor时，
可使用 :ref:`cn_api_fluid_layers_StaticRNN` 。

DynamicRNN可以处理一批序列数据，其中每个样本序列的长度可以不同，每个序列的长度信息记录在LoD里面。
DynamicRNN会按照时间步将输入序列展开，用户可以在 :code:`with` block中定义如何处理每个时间步。

<font color="#FF0000">**注意：目前不支持在DynamicRNN中任何层上配置 is_sparse = True。**</font>


step_input
^^^^^^^^^^^^^^^^^^^^^

.. py:method:: step_input(x, level=0)

将序列标记为动态RNN输入。

参数：
    - **x** (Variable) - 输入序列LoDTensor，代表由长度不同的多个序列组成的minibatch，第一个维度的值等于minibatch内所有序列的长度之和。可以为一个动态RNN类设置多个输入，多个输入LoDTensor必须携带完全相同的LoD信息。因此，多个输入Tensor的第一个维度必须相同，其它维度可以不同。
    - **level** (int) - 用于拆分步骤的LoD层级，取值范围是 :code:`[0, x.lod_level)`，默认值是0。

返回： 输入序列每个时间步的数据。执行第 :code:`step_idx` 个时间步时，若输入 :code:`x` 中有 :code:`num_sequences` 个长度不小于 :code:`step_idx` 的序列，则这个时间步返回值中只包含了这 :code:`num_sequences` 个序列第 :code:`step_idx` 时间步的数据。

返回类型：Variable

抛出异常：
  - :code:`ValueError` ：当 :code:`step_input()` 接口在RNN :code:`block()` 接口外面被调用时。
  - :code:`TypeError`：当输入x类型不是Variable时。


**代码示例**

..  code-block:: python

      import paddle.fluid as fluid

      sentence = fluid.layers.data(name='sentence', shape=[1], dtype='int64', lod_level=1)
      embedding = fluid.layers.embedding(input=sentence, size=[65536, 32], is_sparse=True)

      drnn = fluid.layers.DynamicRNN()
      with drnn.block():
          # 将embedding标记为RNN的输入，每个时间步取句子中的一个字进行处理
          word = drnn.step_input(embedding)
          prev = drnn.memory(shape=[200])
          hidden = fluid.layers.fc(input=[word, prev], size=200, act='relu')
          drnn.update_memory(prev, hidden)  # set prev to hidden
          drnn.output(hidden)

      # 获得RNN最后一个时间步的计算结果
      rnn_output = drnn()
      last = fluid.layers.sequence_last_step(rnn_output)

static_input
^^^^^^^^^^^^^^^^^^^^^

.. py:method:: static_input(x)

将变量标记为RNN的静态输入。静态输入在RNN执行过程中保持不变。DynamicRNN中不是采取padding的方式来支持不等长序列数据的处理，每个时间步中，已经处理完的输入序列不再参与计算，因此需要对静态输入进行收缩处理，移除不再参与计算的序列。

参数:
    - **x** (Variable) - 静态输入序列LoDTensor，要求持有与输入LoDTensor（通过 :code:`step_input` 设置的输入）相同的序列个数。如果输入x的LoD信息为空，则会被当成由 :code:`x.shape[0]` 个长度为1序列组成。

返回: 经过按照RNN输入LoD信息重排序、且收缩处理后的静态输入LoDTensor。执行第 :code:`step_idx` 个时间步时，如果输入序列中只有 :code:`num_sequences` 长度不小于 :code:`step_idx` 的序列，静态输入也会进行收缩处理，只返回对应的 :code:`num_sequences` 个序列对应的数据。

返回类型：Variable

抛出异常：
    - :code:`ValueError`：当 :code:`static_input()` 接口在RNN :code:`block()` 接口外面被调用时。
    - :code:`TypeError`：当输入x类型不是Variable类型时。
    - :code:`RuntimeError`：当 :code:`static_input()` 接口在 :code:`step_input()` 接口之前被调用时。

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid

    sentence = fluid.layers.data(name='sentence', dtype='float32', shape=[32], lod_level=1)
    encoder_proj = fluid.layers.data(name='encoder_proj', dtype='float32', shape=[32], lod_level=1)
    decoder_boot = fluid.layers.data(name='boot', dtype='float32', shape=[10], lod_level=1)

    drnn = fluid.layers.DynamicRNN()
    with drnn.block():
        # 将sentence标记为RNN的输入，每个时间步取句子中的一个字进行处理
        current_word = drnn.step_input(sentence)
        # 将encode_proj标记为RNN的静态输入
        encoder_word = drnn.static_input(encoder_proj)
        hidden_mem = drnn.memory(init=decoder_boot, need_reorder=True)
        fc_1 = fluid.layers.fc(input=encoder_word, size=30)
        fc_2 = fluid.layers.fc(input=current_word, size=30)
        decoder_inputs = fc_1 + fc_2
        h, _, _ = fluid.layers.gru_unit(input=decoder_inputs, hidden=hidden_mem, size=30)
        drnn.update_memory(hidden_mem, h)
        out = fluid.layers.fc(input=h, size=10, bias_attr=True, act='softmax')
        drnn.output(out)

    rnn_output = drnn()


block
^^^^^^^^^^^^^^^^^^^^^

.. py:method:: block()

定义每个时间步执行的操作。 :code:`block` 语句里面定义的算子序列，将会被执行 :code:`max_sequence_len` 次（ :code:`max_sequence_len` 是输入序列中大的序列长度）。

抛出异常：
    - :code:`ValueError`：当RNN :code:`block()` 接口被多次调用时。

memory
^^^^^^^^^^^^^^^^^^^^^

.. py:method:: memory(init=None, shape=None, value=0.0, need_reorder=False, dtype='float32')

为RNN创建一个memory变量，用于缓存分段数据。

参数：
    - **init** (Variable，可选) – 设置memory初始值的LoDTensor。如果init不是None，将使用init来初始化memory，要求持有与输入LoDTensor（通过 :code:`step_input` 设置的输入）相同的序列个数。如果输入init的LoD信息为空，则会被当成由 :code:`init.shape[0]` 个长度为1序列组成。默认值是None。
    - **shape** (list|tuple，可选) – 当init是None时，用来设置memory的维度。注意：shape中不包含batch_size。若设置 :code:`shape=[D1, D2, ...]`，memory Tensor的实际维度为 :code:`[batch_size, D1, D2, ...]`，其中batch_size由输入序列决定。默认值是None。
    - **value** (float，可选) – 当init是None时，用来设置memory的初始值。默认值是0.0。
    - **need_reorder** (bool，可选) – 当init不是None时，用来决定init是否需要重新排序。动态RNN在计算时，会按照输入LoDTensor中序列的长度对输入进行排序，因此当init中的信息与输入序列样本紧密关联时，需要设置 :code:`need_reorder=True`。默认值是False。
    - **dtype** (str|numpy.dtype，可选) – 当init是None是，初始化memory的数据类型。默认值是"float32"。可设置的值有："float32"，"float64"，"int32"，"int64"。

返回：经过收缩处理后的memory LoDTensor。执行第 :code:`step_idx` 个时间步时，如果输入序列中只有 :code:`num_sequences` 长度不小于 :code:`step_idx` 的序列，memory也会进行收缩处理，只返回对应的 :code:`num_sequences` 个序列对应的数据。

返回类型：Variable

抛出异常：
    - :code:`ValueError`：当 :code:`memory()` 接口在RNN :code:`block()` 接口外面被调用时。
    - :code:`TypeError`：当init被设置了，但是不是Variable类型时。
    - :code:`ValueError`：当 :code:`memory()` 接口在 :code:`step_input()` 接口之前被调用时。

**示例代码一**

..  code-block:: python

  import paddle.fluid as fluid

  sentence = fluid.layers.data(name='sentence', shape=[32], dtype='float32', lod_level=1)
  boot_memory = fluid.layers.data(name='boot', shape=[10], dtype='float32', lod_level=1)

  drnn = fluid.layers.DynamicRNN()
  with drnn.block():
      # 将sentence标记为RNN的输入，每个时间步取句子中的一个字进行处理
      word = drnn.step_input(sentence)
      # 使用boot_memory初始化memory，并且需要依据输入序列进行重排序
      memory = drnn.memory(init=boot_memory, need_reorder=True)
      hidden = fluid.layers.fc(input=[word, memory], size=10, act='tanh')
      drnn.update_memory(ex_mem=memory, new_mem=hidden)
      drnn.output(hidden)

  rnn_output = drnn()


**示例代码二**

..  code-block:: python

  import paddle.fluid as fluid

  sentence = fluid.layers.data(name='sentence', dtype='float32', shape=[32], lod_level=1)

  drnn = fluid.layers.DynamicRNN()
  with drnn.block():
      # 将sentence标记为RNN的输入，每个时间步取句子中的一个字进行处理
      word = drnn.step_input(sentence)
      # 将memory初始化为一个值为0的常量Tensor，shape=[batch_size, 10]，其中batch_size由输入sentence决定
      memory = drnn.memory(shape=[10], dtype='float32', value=0)
      hidden = fluid.layers.fc(input=[word, memory], size=10, act='tanh')
      drnn.update_memory(ex_mem=memory, new_mem=hidden)
      drnn.output(hidden)

  rnn_output = drnn()


update_memory
^^^^^^^^^^^^^^^^^^^^^

.. py:method:: update_memory(ex_mem, new_mem)

将内存从 ``ex_mem`` 更新到 ``new_mem`` 。注意， ``ex_mem`` 和 ``new_mem`` 的 ``shape`` 和数据类型必须相同。

参数：
  - **ex_mem** （memory Variable）-  memory 变量（Variable）
  - **new_mem** （memory Variable）- RNN块中生成的平坦变量（plain  variable）

返回：None

output
^^^^^^^^^^^^^^^^^^^^^

.. py:method:: output(*outputs)

标记RNN输出变量。动态RNN可以将多个变量标记为其输出。使用drnn()获得输出序列。

参数:
    - **\*outputs** - 输出变量。

返回:None
