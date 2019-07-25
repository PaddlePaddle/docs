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

  import paddle.fluid as fluid
  emb_dim = 256
  vocab_size = 10000
  hidden_dim = 512

  data = fluid.layers.data(name='x', shape=[1],
                 dtype='int32', lod_level=1)
  emb = fluid.layers.embedding(input=data, size=[vocab_size, emb_dim], is_sparse=True)
     
  forward_proj = fluid.layers.fc(input=emb, size=hidden_dim * 4,
                                 bias_attr=False)
  forward, _ = fluid.layers.dynamic_lstm(
      input=forward_proj, size=hidden_dim * 4, use_peepholes=False)













