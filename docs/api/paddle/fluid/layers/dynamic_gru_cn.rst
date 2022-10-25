.. _cn_api_fluid_layers_dynamic_gru:

dynamic_gru
-------------------------------


.. py:function::  paddle.fluid.layers.dynamic_gru(input, size, param_attr=None, bias_attr=None, is_reverse=False, gate_activation='sigmoid', candidate_activation='tanh', h_0=None, origin_mode=False)





**注意：该OP的输入只能是LoDTensor，如果您需要处理的输入是Tensor类型，请使用StaticRNN（fluid.layers.** :ref:`cn_api_fluid_layers_StaticRNN` **）。**

该OP用于在完整序列上逐个时间步的进行单层Gated Recurrent Unit（GRU）的计算，单个时间步内GRU的计算支持以下两种计算方式：

如果origin_mode为True，则使用的运算公式来自论文
`Learning Phrase Representations using RNN Encoder Decoder for Statistical Machine Translation <https://arxiv.org/pdf/1406.1078.pdf>`_ 。

.. math::
    u_t & = act_g(W_{ux}x_{t} + W_{uh}h_{t-1} + b_u)\\
    r_t & = act_g(W_{rx}x_{t} + W_{rh}h_{t-1} + b_r)\\
    \tilde{h_t} & = act_c(W_{cx}x_{t} + W_{ch}(r_t \odot h_{t-1}) + b_c)\\
    h_t & = u_t \odot h_{t-1} + (1-u_t) \odot \tilde{h_t}


如果origin_mode为False，则使用的运算公式来自论文
`Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling  <https://arxiv.org/pdf/1412.3555.pdf>`_ 。

公式如下：

.. math::
    u_t & = act_g(W_{ux}x_{t} + W_{uh}h_{t-1} + b_u)\\
    r_t & = act_g(W_{rx}x_{t} + W_{rh}h_{t-1} + b_r)\\
    \tilde{h_t} & = act_c(W_{cx}x_{t} + W_{ch}(r_t \odot h_{t-1}) + b_c)\\
    h_t & = (1-u_t) \odot h_{t-1} + u_t \odot \tilde{h_t}


其中，:math:`x_t` 为当前时间步的输入，这个输入并非 ``input``，该OP不包含 :math:`W_{ux}x_{t}, W_{rx}x_{t}, W_{cx}x_{t}` 的计算，**注意** 要在该OP前使用大小为 ``size`` 的3倍的全连接层并将其输出作为 ``input``；
:math:`h_{t-1}` 为前一时间步的隐状态 ``hidden``； :math:`u_t` 、 :math:`r_t` 、 :math:`\tilde{h_t}` 和 :math:`h_t` 分别代表了GRU单元中update gate（更新门）、reset gate（重置门）、candidate hidden（候选隐状态）和隐状态输出；:math:`\odot` 为逐个元素相乘；
:math:`W_{uh}, b_u` 、 :math:`W_{rh}, b_r` 和 :math:`W_{ch}, b_c` 分别代表更新门、重置门和候选隐状态在计算时使用的权重矩阵和偏置。在实现上，三个权重矩阵合并为一个 :math:`[D, D \times 3]` 形状的Tensor存放，三个偏置拼接为一个 :math:`[1, D \times 3]` 形状的Tensor存放，其中 :math:`D` 为隐单元的数目；权重Tensor存放布局为：:math:`W_{uh}` 和 :math:`W_{rh}` 拼接为 :math:`[D, D  \times 2]` 形状位于前半部分，:math:`W_{ch}` 以 :math:`[D, D]` 形状位于后半部分。


参数
::::::::::::

    - **input** (Variable) – LoD level为1的LoDTensor，表示经线性变换后的序列输入，形状为 :math:`[T, D \times 3]`，其中 :math:`T` 表示mini-batch中所有序列长度之和，:math:`D` 为隐状态特征维度的大小。数据类型为float32或float64。
    - **size** (int) – 隐状态特征维度的大小
    - **param_attr** (ParamAttr，可选) – 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **bias_attr** (ParamAttr，可选) - 指定偏置参数属性的对象。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **is_reverse** (bool，可选) – 指明是否按照和输入相反的序列顺序计算，默认为False。
    - **gate_activation** (str，可选) – 公式中 :math:`act_g` 激活函数的类型。支持identity、sigmoid、tanh、relu四种激活函数类型，默认为sigmoid。
    - **candidate_activation** (str，可选) – 公式中 :math:`act_c` 激活函数的类型。支持identity、sigmoid、tanh、relu四种激活函数类型，默认为tanh。
    - **h_0** (Variable，可选) – 表示初始隐状态的Tensor，若未提供，则默认为0。其形状为 :math:`[N, D]`，其中 :math:`N` 为输入mini-batch中序列的数目，:math:`D` 为隐状态特征维度的大小。数据类型与 ``input`` 相同。默认值为None。
    - **origin_mode** (bool，可选) – 指明要使用的GRU计算方式，两种计算方式具体差异见公式描述，默认值为False。

返回
::::::::::::
 形状为 :math:`[T, D]` 、LoD level为1的LoDTensor，其中 :math:`T` 表示mini-batch中所有序列长度之和，:math:`D` 为隐状态特征维度的大小。表示经过GRU变换的输出特征序列，和 ``input`` 具有相同的LoD（序列长度）和数据类型。

返回类型
::::::::::::
 Variable


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.dynamic_gru