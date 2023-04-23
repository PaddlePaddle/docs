.. _cn_api_fluid_layers_gru_unit:

gru_unit
-------------------------------


.. py:function:: paddle.fluid.layers.gru_unit(input, hidden, size, param_attr=None, bias_attr=None, activation='tanh', gate_activation='sigmoid', origin_mode=False)




Gated Recurrent Unit（GRU）循环神经网络计算单元。该OP用于完成单个时间步内GRU的计算，支持以下两种计算方式：

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


其中，:math:`x_t` 为当前时间步的输入，这个输入并非 ``input``，该OP不包含 :math:`W_{ux}x_{t}, W_{rx}x_{t}, W_{cx}x_{t}` 的计算，**注意** 要在该OP前使用大小为GRU隐单元数目的3倍的全连接层并将其输出作为 ``input``；
:math:`h_{t-1}` 为前一时间步的隐状态 ``hidden``； :math:`u_t` 、 :math:`r_t` 、 :math:`\tilde{h_t}` 和 :math:`h_t` 分别代表了GRU单元中update gate（更新门）、reset gate（重置门）、candidate hidden（候选隐状态）和隐状态输出；:math:`\odot` 为逐个元素相乘；
:math:`W_{uh}, b_u` 、 :math:`W_{rh}, b_r` 和 :math:`W_{ch}, b_c` 分别代表更新门、重置门和候选隐状态在计算时使用的权重矩阵和偏置。在实现上，三个权重矩阵合并为一个 :math:`[D, D \times 3]` 形状的Tensor存放，三个偏置拼接为一个 :math:`[1, D \times 3]` 形状的Tensor存放，其中 :math:`D` 为隐单元的数目；权重Tensor存放布局为：:math:`W_{uh}` 和 :math:`W_{rh}` 拼接为 :math:`[D, D  \times 2]` 形状位于前半部分，:math:`W_{ch}` 以 :math:`[D, D]` 形状位于后半部分。


参数
::::::::::::

    - **input** (Variable) – 表示经线性变换后当前时间步的输入，是形状为 :math:`[N, D \times 3]` 的二维Tensor，其中 :math:`N` 为batch_size， :math:`D` 为隐单元的数目。数据类型为float32或float64。
    - **hidden** (Variable) –  表示上一时间步产生的隐状态，是形状为 :math:`[N, D]` 的二维Tensor，其中 :math:`N` 为batch_size， :math:`D` 为隐单元的数目。数据类型与 ``input`` 相同。
    - **size** (integer) – 输入数据 ``input`` 特征维度的大小，需要是隐单元数目的3倍。
    - **param_attr** (ParamAttr，可选) – 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **bias_attr** (ParamAttr，可选) - 指定偏置参数属性的对象。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **activation** (string) –  公式中 :math:`act_c` 激活函数的类型。支持identity、sigmoid、tanh、relu四种激活函数类型，默认为tanh。
    - **gate_activation** (string) – 公式中 :math:`act_g` 激活函数的类型。支持identity、sigmoid、tanh、relu四种激活函数类型，默认为sigmoid。
    - **origin_mode** (bool) – 指明要使用的GRU计算方式，两种计算方式具体差异见公式描述，默认值为False。


返回
::::::::::::
Variable的三元组，包含三个与 ``input`` 相同数据类型的Tensor，分别表示下一时间步的隐状态（ :math:`h_t` ）、重置的前一时间步的隐状态（ :math:`r_t \odot h_{t-1}` ）和 :math:`h_t, r_t, \tilde{h_t}` 的拼接，形状分别为 :math:`[N, D]` 、 :math:`[N, D]` 和 :math:`[N, D \times 3]`。通常只有下一时间步的隐状态（ :math:`h_t` ）作为GRU的输出和隐状态使用，其他内容只是中间计算结果。

返回类型
::::::::::::
 tuple


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.gru_unit