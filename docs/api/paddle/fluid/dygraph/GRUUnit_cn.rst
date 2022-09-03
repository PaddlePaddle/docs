.. _cn_api_fluid_dygraph_GRUUnit:

GRUUnit
-------------------------------

.. py:class:: paddle.fluid.dygraph.GRUUnit(name_scope, size, param_attr=None, bias_attr=None, activation='tanh', gate_activation='sigmoid', origin_mode=False, dtype='float32')




该接口用于构建 ``GRU(Gated Recurrent Unit)`` 类的一个可调用对象，具体用法参照 ``代码示例``。其用于完成单个时间步内GRU的计算，支持以下两种计算方式：

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


其中，:math:`x_t` 为当前时间步的输入，:math:`h_{t-1}` 为前一时间步的隐状态 ``hidden``； :math:`u_t` 、 :math:`r_t` 、 :math:`\tilde{h_t}` 和 :math:`h_t` 分别代表了GRU单元中update gate（更新门）、reset gate（重置门）、candidate hidden（候选隐状态）和隐状态输出；:math:`\odot` 为逐个元素相乘；
:math:`W_{uh}, b_u` 、 :math:`W_{rh}, b_r` 和 :math:`W_{ch}, b_c` 分别代表更新门、重置门和候选隐状态在计算时使用的权重矩阵和偏置。在实现上，三个权重矩阵合并为一个维度为 :math:`[D, D \times 3]` 的Tensor存放。

参数
::::::::::::

    - **size** (int) – 输入数据的维度大小。
    - **param_attr** (ParamAttr，可选) – 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    **注意**
      - 权重参数维度为 :math:`[T, 3×D]` ， :math:`D` 是隐藏状态的规模（hidden size），其值与输入size相关，计算方式为size除以3取整。
      - 权重参数矩阵所有元素由两部分组成，一是update gate和reset gate的权重，维度为 :math:`[D, 2×D]` 的2D Tensor，数据类型可以为float32或float64；二是候选隐藏状态（candidate hidden state）的权重矩阵，维度为 :math:`[D, D]` 的2D Tensor，数据类型可以为float32或float64。
    - **bias_attr** (ParamAttr，可选) - 指定偏置参数属性的对象。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **activation** (str，可选) –  公式中 :math:`act_c` 激活函数的类型。可以为'identity'、'sigmoid'、'tanh'、'relu'四种激活函数设置值。默认值为'tanh'。
    - **gate_activation** (str，可选) – 公式中 :math:`act_g` 激活函数的类型。可以为'identity'、'sigmoid'、'tanh'、'relu'四种激活函数设置值。默认值为'sigmoid'。
    - **origin_mode** (bool) – 指明要使用的GRU计算方式，两种计算方式具体差异见公式描述。默认值为False。
    - **dtype** (str，可选) – 该层的数据类型，可以为'float32', 'float64'。默认值为'float32'。

返回
::::::::::::
 
    None.
    
代码示例
::::::::::::

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.dygraph.base as base
    import numpy

    lod = [[2, 4, 3]]
    D = 5
    T = sum(lod[0])

    input = numpy.random.rand(T, 3 * D).astype('float32')
    hidden_input = numpy.random.rand(T, D).astype('float32')
    with fluid.dygraph.guard():
        x = numpy.random.random((3, 32, 32)).astype('float32')
        gru = fluid.dygraph.GRUUnit(size=D * 3)
        dy_ret = gru(
        base.to_variable(input), base.to_variable(hidden_input))


属性
::::::::::::
属性
::::::::::::
weight
'''''''''

本层的可学习参数，类型为 ``Parameter``

bias
'''''''''

本层的可学习偏置，类型为 ``Parameter``
