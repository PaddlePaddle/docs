.. _cn_api_fluid_layers_dynamic_gru:

dynamic_gru
-------------------------------

.. py:function::  paddle.fluid.layers.dynamic_gru(input, size, param_attr=None, bias_attr=None, is_reverse=False, gate_activation='sigmoid', candidate_activation='tanh', h_0=None, origin_mode=False)



**实现了Gated Recurrent Unit层。**

如果origin_mode为False，那么gru运算公式来自论文 `Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling <https://arxiv.org/abs/1412.3555>`_ 。


公式如下：

.. math::
  u_{t}=act_g(W_{ux}x_{t}+W_{uh}h_{t-1}+b_{u})
.. math::
  r_{t}=act_g(W_{rx}x_{t}+W_{rh}h_{t-1}+b_{r})
.. math::
  \widetilde{h_{t}}=act_{c}(W_{cx}x_{t}+W_{ch}(r_{t}\odot h_{t-1})+b_c)
.. math::
  h_t=(1-u_t)\odot h_{t-1}+u_t\odot \widetilde{h_t}




如果origin_mode为True，那么运算公式来自于 `Learning Phrase Representations using RNN Encoder Decoder for Statistical Machine Translation <https://arxiv.org/pdf/1406.1078.pdf>`_



公式如下:

.. math::
    u_t & = act_g(W_{ux}x_{t} + W_{uh}h_{t-1} + b_u)\\
    r_t & = act_g(W_{rx}x_{t} + W_{rh}h_{t-1} + b_r)\\
    \tilde{h_t} & = act_c(W_{cx}x_{t} + W_{ch}(r_t \odot h_{t-1}) + b_c)\\
    h_t & = u_t \odot h_{t-1} + (1-u_t) \odot \tilde{h_t}




其中， :math:`\odot` 为按元素将向量相乘。 :math:`act_g` 是更新门（update gate）和重置门（reset gate）的激励函数(activation)， 常为 :math:`sigmoid` 函数。 :math:`act_c` 是candidate hidden state(候选隐藏状态)的激励函数，常为 :math:`tanh` 。

注意 :math:`W_{ux}x_{t},W_{rx}x_{t},W_{cx}x_{t}` 这些在 input  :math:`x_t` 上的操作不包括在该运算中。用户可以选择性地在GRU层之前使用FC层来进行这一操作。



参数:
  - **input** (Variable) – dynamic_gru层的输入, 支持variable time length input sequence（可变时长输入序列）。 本变量底层的tensor是一个(T×3D)矩阵， 其中T是该mini-batch中总时间步数， D是隐藏状态的规模（hidden size）。
  - **size** (int) – GRU cell的维度
  - **param_attr** (ParamAttr|None)  –  可学习的隐藏层权重矩阵的参数属性。
    注意：
                                    - 该矩阵为一个（T X 3D）矩阵。其中D为隐藏状态的规模（hidden size）
                                    - 该矩阵的所有元素由两部分组成。一是update gate和reset gate的权重，形为（D X 2D)，二是候选隐藏状态（candidate hidden state）的权重，形为 (D X D)
    如果该函数参数被设为None或者 ``ParamAttr`` 类的属性之一，则会生成一个 ``ParamAttr`` 类的对象作为param_attr。如果param_attr未被初始化（即其构造函数未被设置），Xavier会负责初始化它。 默认值为None。
  - **bias_attr** (ParamAttr|bool|None) - GRU层bias的参数属性。该（1 X 3D）形的bias变量将会连结（concatenate）在update gate（更新门）、reset gate（重置门）、candidate calculations（候选隐藏状态计算）后。如果值为False，将没有bias会应用到上述三个过程中。如果该函数参数被设为None或者 ``ParamAttr`` 类的属性之一， ``dynamic_gru`` 会生成一个 ``ParamAttr`` 类的对象作为param_attr。如果bias_attr未被初始化（即其构造函数未被设置），则它会被初始化为0。默认值为None。
  - **is_reverse** (bool) –是否计算反GRU(reversed GRU)，默认为False
  - **gate_activation** (str) – update gate 和 reset gate的激励函数（activation）。 可选择[“sigmoid”, “tanh”, “relu”, “identity”]其一, 默认为 “sigmoid”
  - **candidate_activation** (str) – candidate hidden state（候选隐藏状态）计算所需的激励函数（activation）。 可从[“sigmoid”, “tanh”, “relu”, “identity”]中选择, 默认为 “tanh”
  - **h_0** (Variable) – 该函数参数为初始隐藏状态。若未赋值，则默认为0。它是一个 (N x D) tensor, 其中 N 为输入mini-batch的总时间步数， D 为 隐藏状态规模(hidden size)


返回： GRU的隐藏状态(hidden state)。形为（T X D），序列长度和输入相同。

返回类型: 变量（variable）


**代码示例**

..  code-block:: python

    import paddle.fluid as fluid

    dict_dim, emb_dim = 128, 64
    data = fluid.layers.data(name='sequence', shape=[1],
                             dtype='int32', lod_level=1)
    emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])
    hidden_dim = 512
    x = fluid.layers.fc(input=emb, size=hidden_dim * 3)
    hidden = fluid.layers.dynamic_gru(input=x, size=hidden_dim)
















