.. _cn_api_fluid_layers_DynamicRNN:

DynamicRNN
-------------------------------

.. py:class:: paddle.fluid.layers.DynamicRNN(name=None)


动态RNN可以处理一批序列数据,每个样本序列的长度可以不同。这个API自动批量处理它们。

必须设置输入lod，请参考 ``lod_tensor``

动态RNN将按照timesteps展开开序列。用户需要在with block中定义如何处理处理每个timestep。

memory用于缓存分段数据。memory的初始值可以是零，也可以是其他变量。

动态RNN可以将多个变量标记为其输出。使用drnn()获得输出序列。

.. note::
    目前不支持在DynamicRNN中任何层上配置 is_sparse = True

**代码示例**

..  code-block:: python

  import paddle.fluid as fluid
  
  sentence = fluid.layers.data(name='sentence', shape=[1], dtype='int64', lod_level=1)
  embedding = fluid.layers.embedding(input=sentence, size=[65536, 32], is_sparse=True)
  
  drnn = fluid.layers.DynamicRNN()
  with drnn.block():
      word = drnn.step_input(embedding)
      prev = drnn.memory(shape=[200])
      hidden = fluid.layers.fc(input=[word, prev], size=200, act='relu')
      drnn.update_memory(prev, hidden)  # set prev to hidden
      drnn.output(hidden)
     
  # 获得上一个timestep的rnn，该值是一个编码后的结果
  rnn_output = drnn()
  last = fluid.layers.sequence_last_step(rnn_output)


.. py:method:: step_input(x, level=0)

    将序列标记为动态RNN输入。

参数:
      - **x** (Variable) - 含lod信息的输入序列
      - **level** (int) - 用于拆分步骤的LOD层级，默认值0

返回:当前的输入序列中的timestep。

.. py:method:: static_input(x)

将变量标记为RNN输入。输入不会分散到timestep中。为可选项。

参数:
      - **x** (Variable) - 输入序列

返回:可以访问的RNN的输入变量。

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
     
    sentence = fluid.layers.data(name='sentence', dtype='float32', shape=[32], lod_level=1)
    encoder_proj = fluid.layers.data(name='encoder_proj', dtype='float32', shape=[32], lod_level=1)
    decoder_boot = fluid.layers.data(name='boot', dtype='float32', shape=[10], lod_level=1)
     
    drnn = fluid.layers.DynamicRNN()
    with drnn.block():
        current_word = drnn.step_input(sentence)
        encoder_word = drnn.static_input(encoder_proj)
        hidden_mem = drnn.memory(init=decoder_boot, need_reorder=True)
        fc_1 = fluid.layers.fc(input=encoder_word, size=30, bias_attr=False)
        fc_2 = fluid.layers.fc(input=current_word, size=30, bias_attr=False)
        decoder_inputs = fc_1 + fc_2
        h, _, _ = fluid.layers.gru_unit(input=decoder_inputs, hidden=hidden_mem, size=30)
        drnn.update_memory(hidden_mem, h)
        out = fluid.layers.fc(input=h, size=10, bias_attr=True, act='softmax')
        drnn.output(out)
     
    rnn_output = drnn()


.. py:method:: block()

用户在RNN中定义operators的block。

.. py:method:: memory(init=None, shape=None, value=0.0, need_reorder=False, dtype='float32')

为动态rnn创建一个memory 变量。

如果 ``init`` 不是None， ``memory`` 将由这个变量初始化。参数 ``need_reorder`` 用于将memory重新排序作为输入变量。当memory初始化依赖于输入样本时，应该将其设置为True。

**代码示例**

..  code-block:: python

  import paddle.fluid as fluid
  
  sentence = fluid.layers.data(name='sentence', shape=[32], dtype='float32', lod_level=1)
  boot_memory = fluid.layers.data(name='boot', shape=[10], dtype='float32', lod_level=1)
  
  drnn = fluid.layers.DynamicRNN()
  with drnn.block():
      word = drnn.step_input(sentence)
      memory = drnn.memory(init=boot_memory, need_reorder=True)
      hidden = fluid.layers.fc(input=[word, memory], size=10, act='tanh')
      drnn.update_memory(ex_mem=memory, new_mem=hidden)
      drnn.output(hidden)

  rnn_output = drnn()



否则，如果已经设置 ``shape`` 、 ``value`` 、 ``dtype`` ，memory将被 ``value`` 初始化

**代码示例**

..  code-block:: python

  import paddle.fluid as fluid

  sentence = fluid.layers.data(name='sentence', dtype='float32', shape=[32], lod_level=1)

  drnn = fluid.layers.DynamicRNN()
  with drnn.block():
      word = drnn.step_input(sentence)
      memory = drnn.memory(shape=[10], dtype='float32', value=0)
      hidden = fluid.layers.fc(input=[word, memory], size=10, act='tanh')
      drnn.update_memory(ex_mem=memory, new_mem=hidden)
      drnn.output(hidden)

  rnn_output = drnn()


参数：
    - **init** (Variable|None) – 初始化的Variable
    - **shape** (list|tuple) – memory shape，形状不包含batch_size
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

      

  




