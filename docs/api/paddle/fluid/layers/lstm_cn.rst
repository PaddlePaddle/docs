.. _cn_api_fluid_layers_lstm:

lstm
-------------------------------


.. py:function::  paddle.fluid.layers.lstm(input, init_h, init_c, max_len, hidden_size, num_layers, dropout_prob=0.0, is_bidirec=False, is_test=False, name=None, default_initializer=None, seed=-1)




.. note::
    该 OP 仅支持 GPU 设备运行

该 OP 实现了 LSTM，即 Long-Short Term Memory（长短期记忆）运算 - `Hochreiter, S., & Schmidhuber, J. (1997) <https://www.bioinf.jku.at/publications/older/2604.pdf>`_ 。

该 OP 的实现不包括 diagonal/peephole 连接，参见 `Gers, F. A., & Schmidhuber, J. (2000) <ftp://ftp.idsia.ch/pub/juergen/TimeCount-IJCNN2000.pdf>`_ 。
如果需要使用 peephole 连接方法，请使用 :ref:`cn_api_fluid_layers_dynamic_lstm` 。

该 OP 对于序列中每一个时间步的计算公式如下：

.. math::
  i_t = \sigma(W_{ix}x_{t} + W_{ih}h_{t-1} + b_{x_i} + b_{h_i})
.. math::
  f_t = \sigma(W_{fx}x_{t} + W_{fh}h_{t-1} + b_{x_f} + b_{h_f})
.. math::
  o_t = \sigma(W_{ox}x_{t} + W_{oh}h_{t-1} + b_{x_o} + b_{h_o})
.. math::
  \widetilde{c_t} = tanh(W_{cx}x_t + W_{ch}h_{t-1} + b{x_c} + b_{h_c})
.. math::
  c_t = f_t \odot c_{t-1} + i_t \odot \widetilde{c_t}
.. math::
  h_t = o_t \odot tanh(c_t)

公式中的概念信息如下：
      - :math:`x_{t}` 表示时间步 :math:`t` 的输入
      - :math:`h_{t}` 表示时间步 :math:`t` 的 hidden 状态
      - :math:`h_{t-1}, c_{t-1}` 分别表示前一个时间步的 hidden 和 cell 状态
      - :math:`\widetilde{c_t}` 表示候选的 cell 状态
      - :math:`i_t` ，:math:`f_t` 和 :math:`o_t` 分别为 input gate，forget gate，output gate
      - :math:`W` 表示 weight （例如，:math:`W_{ix}` 是在计算 input gate :math:`i_t` 时，对输入 :math:`x_{t}` 做线性变换的 weight）
      - :math:`b` 表示 bias （例如，:math:`b_{i}` 是 input gate 的 bias）
      - :math:`\sigma` 表示 gate 的非线性激活函数，默认为 sigmoid
      - :math:`\odot` 表示矩阵的 Hadamard product，即对两个维度相同的矩阵，将相同位置的元素相乘，得到另一个维度相同的矩阵

参数
::::::::::::

  - **input** ( :ref:`api_guide_Variable` ) - LSTM 的输入 Tensor，维度为 :math:`[batch\_size, seq\_len, input\_dim]` 的 3-D Tensor，其中 seq_len 为序列的长度，input_dim 为序列词嵌入的维度。数据类型为 float32 或者 float64。
  - **init_h** ( :ref:`api_guide_Variable` ) – LSTM 的初始 hidden 状态，维度为 :math:`[num\_layers, batch\_size, hidden\_size]` 的 3-D Tensor，其中 num_layers 是LSTM 的总层数，hidden_size 是隐层维度。如果 is_bidirec = True，维度应该为 :math:`[num\_layers*2, batch\_size, hidden\_size]`。数据类型为 float32 或者 float64。
  - **init_c** ( :ref:`api_guide_Variable` ) - LSTM 的初始 cell 状态。维度为 :math:`[num\_layers, batch\_size, hidden\_size]` 的 3-D Tensor，其中 num_layers 是LSTM 的总层数，hidden_size 是隐层维度。如果 is_bidirec = True，维度应该为 :math:`[num\_layers*2, batch\_size, hidden\_size]`。数据类型为 float32 或者 float64。
  - **max_len** (int) – LSTM 的最大长度。输入 Tensor 的第一个 input_dim 不能大于 max_len。
  - **hidden_size** (int) - LSTM hidden 状态的维度。
  - **num_layers** (int) –  LSTM 的总层数。例如，该参数设置为 2，则会堆叠两个 LSTM，其第一个 LSTM 的输出会作为第二个 LSTM 的输入。
  - **dropout_prob** (float，可选) – dropout 比例，dropout 只在 rnn 层之间工作，而不是在时间步骤之间。dropout 不作用于最后的 rnn 层的 rnn 输出中。默认值为 0.0。
  - **is_bidirec** (bool，可选) – 是否是双向的 LSTM。默认值为 False。
  - **is_test** (bool，可选) – 是否在测试阶段。默认值为 False。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
  - **default_initializer** (Initializer，可选) – 用于初始化权重的初始化器，如果为 None，将进行默认初始化。默认值为 None。
  - **seed** (int，可选) – LSTM 中 dropout 的 seed，如果是-1，dropout 将使用随机 seed。默认值为 1。

返回
::::::::::::
 经过 lstm 运算输出的三个 Tensor 的 tuple，包括

- rnn_out：LSTM hidden 的输出结果的 Tensor，数据类型与 input 一致，维度为 :math:`[batch\_size, seq\_len, hidden\_size]`。如果 ``is_bidirec`` 设置为 True，则维度为 :math:`[batch\_size, seq\_len, hidden\_size*2]`
- last_h：LSTM 最后一步的 hidden 状态的 Tensor，数据类型与 input 一致，维度为 :math:`[num\_layers, batch\_size, hidden\_size]`。如果 ``is_bidirec`` 设置为 True，则维度为 :math:`[num\_layers*2, batch\_size, hidden\_size]`
- last_c：LSTM 最后一步的 cell 状态的 Tensor，数据类型与 input 一致，维度为 :math:`[num\_layers, batch\_size, hidden\_size]`。如果 ``is_bidirec`` 设置为 True，则维度为 :math:`[num\_layers*2, batch\_size, hidden\_size]`

返回类型
::::::::::::
  tuple（ :ref:`api_guide_Variable` , :ref:`api_guide_Variable` , :ref:`api_guide_Variable` ）

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.lstm
