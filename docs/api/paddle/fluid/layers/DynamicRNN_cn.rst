.. _cn_api_fluid_layers_DynamicRNN:

DynamicRNN
===================


.. py:class:: paddle.fluid.layers.DynamicRNN(name=None)




**注意：该类型的输入仅支持 LoDTensor，如果您需要处理的输入数据是 Tensor 类型，
请使用 StaticRNN（ fluid.layers.** :ref:`cn_api_fluid_layers_StaticRNN` **)。**

DynamicRNN 可以处理一批序列数据，其中每个样本序列的长度可以不同，每个序列的长度信息记录在 LoD 里面。
DynamicRNN 会按照时间步 (time step) 将输入序列展开，用户可以在 :code:`block` 中定义每个时间步要进行的运算。
由于每个输入样本的序列长度不相同，RNN 执行的 step 数由最长的序列决定。
DynamicRNN 的实现采用非 padding 的方式，每个时间步都会对输入数据进行收缩处理，移除已经处理完的序列的信息。
因此，随着时间步的增加，每个时间步处理的样本数（batch size）会逐渐减少。

.. warning::
  目前不支持在 DynamicRNN 的 :code:`block` 中任何层上配置 :code:`is_sparse = True` 。

参数
::::::::::::

    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

成员函数列表：
    - :ref:`cn_api_fluid_layers_DynamicRNN_step_input`，设置输入变量
    - :ref:`cn_api_fluid_layers_DynamicRNN_static_input`，设置静态输入变量
    - :ref:`cn_api_fluid_layers_DynamicRNN_block`，定义每个时间步执行的运算
    - :ref:`cn_api_fluid_layers_DynamicRNN_memory`，创建用于在时间步之间传递信息的变量
    - :ref:`cn_api_fluid_layers_DynamicRNN_update_memory`，更新需要传递的时间步信息
    - :ref:`cn_api_fluid_layers_DynamicRNN_output`，设置时间步的输出变量
    - :ref:`cn_api_fluid_layers_DynamicRNN_call`，获取 RNN 的输出序列


.. _cn_api_fluid_layers_DynamicRNN_step_input:

成员函数 step_input
---------------------------------

方法
::::::::::::
step_input(x, level=0)
'''''''''

将序列 x 设置为 DynamicRNN 输入。输入序列中最长的序列长度，将决定了 RNN 运算的长度。
必须至少为 DynamicRNN 设置一个输入，也可以设置多个输入。
如果多个输入 x 的 :code:`x.lod_level` 都为 1，则要求多个输入 LoDTensor 携带完全相同的 LoD 信息。
当输入 x 的 :code:`x.lod_level >= 2` 时，输入序列将按指定 level 进行展开，每个时间步携带 :code:`x.lod_level - level - 1` 层 LoD 信息，
此时要求多个输入序列的 LoD 在指定 level 上的信息完全一样。

- 示例 1

.. code-block:: text

    # 输入，其中 Si 代表维度为[1, N]的数据
    level = 0
    x.lod = [[2, 1, 3]]
    x.shape = [6, N]
    x.data = [[S0],
              [S0],
              [S1],
              [S2],
              [S2],
              [S2]]

    # 输出
    # step 0，持有 3 个序列的 time step 数据
    out.lod = [[]]
    out.shape = [3, N]
    out.data = [[S2],
                [S0],
                [S1]]

    # step 1，持有 2 个序列的 time step 数据
    out.lod = [[]]
    out.shape = [2, N]
    out.data = [[S2],
                [S0]]

    # step 2，持有 1 个序列的 time step 数据
    out.lod = [[]]
    out.shape = [1, N]
    out.data = [[S2]]


**参数**

    - **x** (Variable) - 输入序列 LoDTensor，代表由长度不同的多个序列组成的 minibatch，要求 :code:`x.lod_level >= 1`。输入 x 第一个维度的值等于 minibatch 内所有序列的长度之和。RNN 有多个输入序列时，多个输入 LoDTensor 的第一个维度必须相同，其它维度可以不同。支持的数据类型有：bool，float16，float32，float64，int8，int16，int32，int64，uint8。
    - **level** (int，可选) - 用于拆分输入序列的 LoD 层级，取值范围是 :math:`[0, x.lod\_level)`，默认值是 0。

**返回**
 输入序列每个时间步的数据。执行第 :code:`step_idx` 个时间步时，若输入 :code:`x` 中有 :code:`num_sequences` 个长度不小于 :code:`step_idx` 的序列，则这个时间步返回值中只包含了这 :code:`num_sequences` 个序列第 :code:`step_idx` 时间步的数据。数据类型和输入一致。如果 :code:`x.lod_level == 1`，返回值的维度是 :math:`\{num\_sequences, x.shape[1], ...\}`。否则，返回值也是一个变长的 LoDTensor。

**返回类型**
Variable

**抛出异常**

    - :code:`ValueError`：当 :code:`step_input()` 接口在 RNN :code:`block()` 接口外面被调用时。
    - :code:`TypeError`：当输入 x 类型不是 Variable 时。


**代码示例**

..  code-block:: python

      import paddle.fluid as fluid

      sentence = fluid.data(name='sentence', shape=[None, 1], dtype='int64', lod_level=1)
      embedding = fluid.layers.embedding(input=sentence, size=[65536, 32], is_sparse=True)

      drnn = fluid.layers.DynamicRNN()
      with drnn.block():
          # 将 embedding 标记为 RNN 的输入，每个时间步取句子中的一个字进行处理
          word = drnn.step_input(embedding)
          # 将 memory 初始化为一个值为 0 的常量 Tensor，shape=[batch_size, 200]，其中 batch_size 由输入 embedding 决定
          memory = drnn.memory(shape=[200])
          hidden = fluid.layers.fc(input=[word, memory], size=200, act='relu')
          # 用 hidden 更新 memory
          drnn.update_memory(ex_mem=memory, new_mem=hidden)
          # 将 hidden 标记为 RNN 的输出
          drnn.output(hidden)

      # 获得 RNN 的计算结果
      rnn_output = drnn()


.. _cn_api_fluid_layers_DynamicRNN_static_input:

成员函数 static_input
---------------------------------

static_input(x)
'''''''''

将变量设置为 RNN 的静态输入。

- 示例 1，静态输入携带 LoD 信息

.. code-block:: text

    # RNN 的输入见 step_input 中的示例
    # 静态输入，其中 Si 代表维度为[1, M]的数据
    x.lod = [[3, 1, 2]]
    x.shape = [6, M]
    x.data = [[S0],
              [S0],
              [S0],
              [S1],
              [S2],
              [S2]]

    # step 0，持有 3 个序列对应的数据
    out.lod = [[2, 3, 1]]
    out.shape = [6, M]
    out.data = [[S2],
                [S2],
                [S0],
                [S0],
                [S0],
                [S1]]

    # step 1，持有 2 个序列对应的数据
    out.lod = [[2, 3]]
    out.shape = [5, M]
    out.data = [[S2],
                [S2],
                [S0],
                [S0],
                [S0]]

    # step 2，持有 1 个序列对应的数据
    out.lod = [[2]]
    out.shape = [2, M]
    out.data = [[S2],
                [S2]]


- 示例 2，静态输入不携带 LoD 信息

.. code-block:: text

    # RNN 的输入见 step_input 中的示例
    # 静态输入，其中 Si 代表维度为[1, M]的数据
    x.lod = [[]]
    x.shape = [3, M]
    x.data = [[S0],
              [S1],
              [S2]]

    # step 0，持有 3 个序列对应的数据
    out.lod = [[]]
    out.shape = [3, M]
    out.data = [[S2],
                [S0],
                [S1]]

    # step 1，持有 2 个序列对应的数据
    out.lod = [[]]
    out.shape = [2, M]
    out.data = [[S2],
                [S0]]

    # step 2，持有 1 个序列对应的数据
    out.lod = [[]]
    out.shape = [1, M]
    out.data = [[S2]]


**参数**

    - **x** (Variable) - 静态输入序列 LoDTensor，要求持有与输入 LoDTensor（通过 :code:`step_input` 设置的输入）相同的序列个数。如果输入 x 的 LoD 信息为空，则会被当成由 :code:`x.shape[0]` 个长度为 1 序列组成。支持的数据类型有：bool，float16，float32，float64，int8，int16，int32，int64，uint8。

**返回**
 经过按照 RNN 输入 LoD 信息重排序、且收缩处理后的静态输入 LoDTensor。执行第 :code:`step_idx` 个时间步时，如果输入序列中只有 :code:`num_sequences` 长度不小于 :code:`step_idx` 的序列，静态输入也会进行收缩处理，只返回对应的 :code:`num_sequences` 个序列对应的数据。数据类型和输入一致。如果 :code:`x.lod == None`，返回值的维度是 :math:`\{num\_sequences, x.shape[1], ...\}`。否则，返回值是一个变长的 LoDTensor。

**返回类型**
Variable

**抛出异常**

    - :code:`ValueError`：当 :code:`static_input()` 接口在 RNN :code:`block()` 接口外面被调用时。
    - :code:`TypeError`：当输入 x 类型不是 Variable 类型时。
    - :code:`RuntimeError`：当 :code:`static_input()` 接口在 :code:`step_input()` 接口之前被调用时。

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid

    sentence = fluid.data(name='sentence', shape=[None, 32], dtype='float32', lod_level=1)
    encoder_proj = fluid.data(name='encoder_proj', shape=[None, 32], dtype='float32', lod_level=1)
    decoder_boot = fluid.data(name='boot', shape=[None, 10], dtype='float32')

    drnn = fluid.layers.DynamicRNN()
    with drnn.block():
        # 将 sentence 标记为 RNN 的输入，每个时间步取句子中的一个字进行处理
        current_word = drnn.step_input(sentence)
        # 将 encode_proj 标记为 RNN 的静态输入
        encoder_word = drnn.static_input(encoder_proj)
        # 使用 boot_memory 初始化 memory，并且需要依据输入序列进行重排序
        memory = drnn.memory(init=decoder_boot, need_reorder=True)
        fc_1 = fluid.layers.fc(input=encoder_word, size=30)
        fc_2 = fluid.layers.fc(input=current_word, size=30)
        decoder_inputs = fc_1 + fc_2
        hidden, _, _ = fluid.layers.gru_unit(input=decoder_inputs, hidden=memory, size=30)
        # 用 hidden 更新 memory
        drnn.update_memory(ex_mem=memory, new_mem=hidden)
        out = fluid.layers.fc(input=hidden, size=10, bias_attr=True, act='softmax')
        # 将 out 标记为 RNN 的输出
        drnn.output(out)

    # 获得 RNN 的计算结果
    rnn_output = drnn()


.. _cn_api_fluid_layers_DynamicRNN_block:

成员函数 block
---------------------------------

block()
'''''''''

定义每个时间步执行的操作。:code:`block` 语句里面定义的算子序列，将会被执行 :code:`max_sequence_len` 次（ :code:`max_sequence_len` 是输入序列中大的序列长度）。

**抛出异常**

    - :code:`ValueError`：当 RNN :code:`block()` 接口被多次调用时。


.. _cn_api_fluid_layers_DynamicRNN_memory:

成员函数 memory
---------------------------------

memory(init=None, shape=None, value=0.0, need_reorder=False, dtype='float32')
'''''''''

为 RNN 创建一个 memory 变量，用于在时间步之间传递信息。
它可以用一个已有的 Tensor 来初始化，也可以初始化为一个特定维度的常量 Tensor。

**参数**

    - **init** (Variable，可选) – 设置 memory 初始值的 LoDTensor。如果 init 不是 None，将使用 init 来初始化 memory，要求持有与输入 LoDTensor（通过 :code:`step_input` 设置的输入）相同的序列个数。如果输入 init 的 LoD 信息为空，则会被当成由 :code:`init.shape[0]` 个长度为 1 序列组成。默认值是 None。
    - **shape** (list|tuple，可选) – 当 init 是 None 时，用来设置 memory 的维度。注意：shape 中不包含 batch_size。若设置 :math:`shape=\{D_1, D_2, ...\}`，memory Tensor 的实际维度为 :math:`\{batch\_size, D_1, D_2, ...\}`，其中 batch_size 由输入序列决定。默认值是 None。
    - **value** (float，可选) – 当 init 是 None 时，用来设置 memory 的初始值。默认值是 0.0。
    - **need_reorder** (bool，可选) – 当 init 不是 None 时，用来决定 init 是否需要重新排序。动态 RNN 在计算时，会按照输入 LoDTensor 中序列的长度对输入进行排序，因此当 init 中的信息与输入序列样本紧密关联时，需要设置 :code:`need_reorder=True`。默认值是 False。
    - **dtype** (str|numpy.dtype，可选) – 当 init 是 None 是，初始化 memory 的数据类型。默认值是"float32"。可设置的字符串值有："float32"，"float64"，"int32"，"int64"。

**返回**
经过收缩处理后的 memory LoDTensor。执行第 :code:`step_idx` 个时间步时，如果输入序列中只有 :code:`num_sequences` 长度不小于 :code:`step_idx` 的序列，memory 也会进行收缩处理，只返回对应的 :code:`num_sequences` 个序列对应的数据。

**返回类型**
Variable

**抛出异常**

    - :code:`ValueError`：当 :code:`memory()` 接口在 RNN :code:`block()` 接口外面被调用时。
    - :code:`TypeError`：当 init 被设置了，但是不是 Variable 类型时。
    - :code:`ValueError`：当 :code:`memory()` 接口在 :code:`step_input()` 接口之前被调用时。

代码示例一
::::::::::::

..  code-block:: python

    import paddle.fluid as fluid

    sentence = fluid.data(name='sentence', shape=[None, 32], dtype='float32', lod_level=1)
    boot_memory = fluid.data(name='boot', shape=[None, 10], dtype='float32')

    drnn = fluid.layers.DynamicRNN()
    with drnn.block():
        # 将 sentence 标记为 RNN 的输入，每个时间步取句子中的一个字进行处理
        word = drnn.step_input(sentence)
        # 使用 boot_memory 初始化 memory，并且需要依据输入序列进行重排序
        memory = drnn.memory(init=boot_memory, need_reorder=True)
        hidden = fluid.layers.fc(input=[word, memory], size=10, act='tanh')
        # 用 hidden 更新 memory
        drnn.update_memory(ex_mem=memory, new_mem=hidden)
        # 将 hidden 标记为 RNN 的输出
        drnn.output(hidden)

    # 获得 RNN 的计算结果
    rnn_output = drnn()


代码示例二
::::::::::::

..  code-block:: python

    import paddle.fluid as fluid

    sentence = fluid.data(name='sentence', shape=[None, 32], dtype='float32', lod_level=1)

    drnn = fluid.layers.DynamicRNN()
    with drnn.block():
        # 将 sentence 标记为 RNN 的输入，每个时间步取句子中的一个字进行处理
        word = drnn.step_input(sentence)
        # 将 memory 初始化为一个值为 0 的常量 Tensor，shape=[batch_size, 10]，其中 batch_size 由输入 sentence 决定
        memory = drnn.memory(shape=[10], dtype='float32', value=0)
        hidden = fluid.layers.fc(input=[word, memory], size=10, act='tanh')
        # 用 hidden 更新 memory
        drnn.update_memory(ex_mem=memory, new_mem=hidden)
        # 将 hidden 标记为 RNN 的输出
        drnn.output(hidden)

    # 获得 RNN 的计算结果
    rnn_output = drnn()


.. _cn_api_fluid_layers_DynamicRNN_update_memory:

成员函数 update_memory
---------------------------------

update_memory(ex_mem, new_mem)
'''''''''

将需要在时间步之间传递的信息更新。

**参数**

  - **ex_mem** (Variable) - 上一个时间步的信息。
  - **new_mem** (Variable) - 新的时间步信息。:code:`new_mem` 的维度和数据类型必须与 :code:`ex_mem` 一致。

**返回**
无

**抛出异常**

    - :code:`ValueError`：当 :code:`update_memory()` 接口在 RNN :code:`block()` 接口外面被调用时。
    - :code:`TypeError`：当 :code:`ex_mem` 或 :code:`new_mem` 不是 Variable 类型时。
    - :code:`ValueError`：当 :code:`ex_mem` 不是使用 :code:`memory()` 接口定义的 memory 时。
    - :code:`ValueError`：当 :code:`update_memory()` 接口在 :code:`step_input()` 接口之前被调用时。


.. _cn_api_fluid_layers_DynamicRNN_output:

成员函数 output
---------------------------------

output(*outputs)
'''''''''

设置 outputs 为 RNN 每个时间步的输出变量。

**参数**

    - **\*outputs** (Variable ...) - 输出 Tensor，可同时将多个 Variable 标记为输出。

**返回**
无

**抛出异常**

    - :code:`ValueError`：当 :code:`output()` 接口在 RNN :code:`block()` 接口外面被调用时。


.. _cn_api_fluid_layers_DynamicRNN_call:

成员函数 __call__
---------------------------------

__call__()
'''''''''

获取 RNN 计算的输出序列。

若定义了 :code:`drnn = DynamicRNN()`，则可以调用 :code:`drnn()` 获得输出序列，该输出序列是通过将每一个时间步的 output 数据合并得到的一个 LoDTensor。
当 RNN 的输入 x（通过 :code:`step_input()` 接口设置）的 :code:`x.lod_level` 为 1 时，该输出 LoDTensor 将会和输入 x 持有完全相同的 LoD 信息。
通过 :code:`drnn()` 获取的 RNN 输出 LoDTensor 中包含了所有时间步的计算结果，可调用 :ref:`cn_api_fluid_layers_sequence_last_step` 获取最后一个时间步的计算结果。

**参数**

    无

**返回**
RNN 的输出序列。

**返回类型**
Variable 或 Variable list

**抛出异常**

    - :code:`ValueError`：当 :code:`__call__()` 接口在 RNN :code:`block()` 定义之前被调用时。

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid

    sentence = fluid.data(name='sentence', shape=[None, 32], dtype='float32', lod_level=1)
    encoder_proj = fluid.data(name='encoder_proj', shape=[None, 32], dtype='float32', lod_level=1)
    decoder_boot = fluid.data(name='boot', shape=[None, 10], dtype='float32')

    drnn = fluid.layers.DynamicRNN()
    with drnn.block():
        # 将 sentence 标记为 RNN 的输入，每个时间步取句子中的一个字进行处理
        current_word = drnn.step_input(sentence)
        # 将 encode_proj 标记为 RNN 的静态输入
        encoder_word = drnn.static_input(encoder_proj)
        # 使用 boot_memory 初始化 memory，并且需要依据输入序列进行重排序
        memory = drnn.memory(init=decoder_boot, need_reorder=True)
        fc_1 = fluid.layers.fc(input=encoder_word, size=30)
        fc_2 = fluid.layers.fc(input=current_word, size=30)
        decoder_inputs = fc_1 + fc_2
        hidden, _, _ = fluid.layers.gru_unit(input=decoder_inputs, hidden=memory, size=30)
        # 用 hidden 更新 memory
        drnn.update_memory(ex_mem=memory, new_mem=hidden)
        out = fluid.layers.fc(input=hidden, size=10, bias_attr=True, act='softmax')
        # 将 hidden 和 out 标记为 RNN 的输出
        drnn.output(hidden, out)

    # 获得 RNN 的计算结果
    hidden, out = drnn()
    # 提取 RNN 最后一个时间步的计算结果
    last = fluid.layers.sequence_last_step(out)
