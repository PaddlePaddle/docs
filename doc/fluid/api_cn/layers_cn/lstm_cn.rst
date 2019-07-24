.. _cn_api_fluid_layers_lstm:

lstm
-------------------------------

.. py:function::  paddle.fluid.layers.lstm(input, init_h, init_c, max_len, hidden_size, num_layers, dropout_prob=0.0, is_bidirec=False, is_test=False, name=None, default_initializer=None, seed=-1)

如果您的设备是GPU，本op将使用cudnn LSTM实现

一个没有 peephole 连接的四门长短期记忆网络。在前向传播中，给定迭代的输出ht和单元输出ct可由递归输入ht-1、单元输入ct-1和上一层输入xt计算，给定矩阵W、R和bias bW, bR由下式计算:

.. math::

  i_t &= \sigma(W_{ix}x_{t} + W_{ih}h_{t-1} + bx_i + bh_i)\\
  f_t &= \sigma(W_{fx}x_{t} + W_{fh}h_{t-1} + bx_f + bh_f)\\
  o_t &= \sigma(W_{ox}x_{t} + W_{oh}h_{t-1} + bx_o + bh_o)\\
  \tilde{c_t} &= tanh(W_{cx}x_t + W_{ch}h_{t-1} + bx_c + bh_c)\\
  c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c_t}\\
  h_t &= o_t \odot tanh(c_t)

公式中：
  - W 项表示权重矩阵(e.g. :math:`W_{ix}` 是从输入门到输入的权重矩阵)
  - b 项表示偏差向量( :math:`b_{xi}` 和 :math:`b_{hi}` 是输入门的偏差向量)
  - sigmoid 是 logistic sigmoid 函数
  - i、f、o、c 分别为输入门、遗忘门、输出门和激活向量，它们的大小与 cell 输出激活向量h相同。
  - :math:`\odot` 是向量的元素乘积
  - tanh是激活函数
  - :math:`\tilde{c_t}` 也称为候选隐藏状态，它是根据当前输入和之前的隐藏状态来计算的

sigmoid的计算公式为： :math:`sigmoid(x) = 1 / (1 + e^{-x})` 。


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

返回：   三个张量， rnn_out, last_h, last_c:

- rnn_out为LSTM hidden的输出结果。形为(seq_len x batch_size x hidden_size)如果is_bidirec设置为True,则形为(seq_len x batch_sze hidden_size * 2)
- last_h(Tensor):  LSTM最后一步的隐藏状态，形状为(num_layers x batch_size x hidden_size)；如果is_bidirec设置为True，形状为(num_layers*2 x batch_size x hidden_size)
- last_c(Tensor)： LSTM最后一步的cell状态，形状为(num_layers x batch_size x hidden_size)；如果is_bidirec设置为True，形状为(num_layers*2 x batch_size x hidden_size)

返回类型:   rnn_out(Tensor),last_h(Tensor),last_c(Tensor)

**代码示例：**

.. code-block:: python

  import paddle.fluid as fluid
  emb_dim = 256
  vocab_size = 10000
  data = fluid.layers.data(name='x', shape=[-1, 100, 1],
                 dtype='int32')
  emb = fluid.layers.embedding(input=data, size=[vocab_size, emb_dim], is_sparse=True)
  batch_size = 20
  max_len = 100
  dropout_prob = 0.2
  input_size = 100
  hidden_size = 150
  num_layers = 1
  init_h = layers.fill_constant( [num_layers, batch_size, hidden_size], 'float32', 0.0 )
  init_c = layers.fill_constant( [num_layers, batch_size, hidden_size], 'float32', 0.0 )

  rnn_out, last_h, last_c = fluid.layers.lstm(emb, init_h, init_c, max_len, hidden_size, num_layers, dropout_prob=dropout_prob)












