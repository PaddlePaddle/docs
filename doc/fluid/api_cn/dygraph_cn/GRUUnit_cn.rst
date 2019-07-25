.. _cn_api_fluid_dygraph_GRUUnit:

GRUUnit
-------------------------------

.. py:class:: paddle.fluid.dygraph.GRUUnit(name_scope, size, param_attr=None, bias_attr=None, activation='tanh', gate_activation='sigmoid', origin_mode=False, dtype='float32')

GRU单元层。GRU执行步骤基于如下等式：


如果origin_mode为True，则该运算公式来自论文
`Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling  <https://arxiv.org/pdf/1412.3555.pdf>`_ 。

公式如下:

.. math::
    u_t=actGate(xu_t+W_{u}h_{t-1}+b_u)
.. math::
    r_t=actGate(xr_t+W_{r}h_{t-1}+b_r)
.. math::
    m_t=actNode(xm_t+W_{c}dot(r_t,h_{t-1})+b_m)
.. math::
    h_t=dot((1-u_t),m_t)+dot(u_t,h_{t-1})


如果origin_mode为False，则该运算公式来自论文
`Learning Phrase Representations using RNN Encoder Decoder for Statistical Machine Translation <https://arxiv.org/pdf/1406.1078.pdf>`_ 。

.. math::
    u_t & = actGate(xu_{t} + W_u h_{t-1} + b_u)\\
    r_t & = actGate(xr_{t} + W_r h_{t-1} + b_r)\\
    m_t & = actNode(xm_t + W_c dot(r_t, h_{t-1}) + b_m)\\
    h_t & = dot((1-u_t), h_{t-1}) + dot(u_t, m_t)


GRU单元的输入包括 :math:`z_t` ， :math:`h_{t-1}` 。在上述等式中， :math:`z_t` 会被分割成三部分： :math:`xu_t` 、 :math:`xr_t` 和 :math:`xm_t`  。
这意味着要为一批输入实现一个全GRU层，我们需要采用一个全连接层，才能得到 :math:`z_t=W_{fc}x_t` 。
:math:`u_t` 和 :math:`r_t` 分别代表了GRU神经元的update gates（更新门）和reset gates(重置门)。
和LSTM不同，GRU少了一个门（它没有LSTM的forget gate）。但是它有一个叫做中间候选隐藏状态（intermediate candidate hidden output）的输出，
记为 :math:`m_t` 。 该层有三个输出： :math:`h_t, dot(r_t,h_{t-1})` 以及 :math:`u_t，r_t，m_t` 的连结(concatenation)。




参数:
    - **name_scope** (str) – 该类的名称
    - **size** (int) – 输入数据的维度
    - **param_attr** (ParamAttr|None) – 可学习的隐藏层权重矩阵的参数属性。
    注意：
      - 该权重矩阵形为 :math:`(T×3D)` ， :math:`D` 是隐藏状态的规模（hidden size）
      - 该权重矩阵的所有元素由两部分组成， 一是update gate和reset gate的权重，形为 :math:`(D×2D)` ；二是候选隐藏状态（candidate hidden state）的权重矩阵，形为 :math:`(D×D)`
      如果该函数参数值为None或者 ``ParamAttr`` 类中的属性之一，gru_unit则会创建一个 ``ParamAttr`` 类的对象作为param_attr。如果param_attr没有被初始化，那么会由Xavier来初始化它。默认值为None
    - **bias_attr** (ParamAttr|bool|None) - GRU的bias变量的参数属性。形为 :math:`(1x3D)` 的bias连结（concatenate）在update gates（更新门），reset gates(重置门)以及candidate calculations（候选隐藏状态计算）中的bias。如果值为False，那么上述三者将没有bias参与运算。若值为None或者 ``ParamAttr`` 类中的属性之一，gru_unit则会创建一个 ``ParamAttr`` 类的对象作为 bias_attr。如果bias_attr没有被初始化，那它会被默认初始化为0。默认值为None。
    - **activation** (str) –  神经元 “actNode” 的激励函数（activation）类型。默认类型为‘tanh’
    - **gate_activation** (str) – 门 “actGate” 的激励函数（activation）类型。 默认类型为 ‘sigmoid’。
    - **dtype** (str) – 该层的数据类型，默认为‘float32’。


返回：  hidden value（隐藏状态的值），reset-hidden value(重置隐藏状态值)，gate values(门值)

返回类型:  元组（tuple）

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.dygraph.base as base
    import numpy

    lod = [[2, 4, 3]]
    D = 5
    T = sum(lod[0])

    hidden_input = numpy.random.rand(T, D).astype('float32')
    with fluid.dygraph.guard():
        x = numpy.random.random((3, 32, 32)).astype('float32')
        gru = fluid.dygraph.GRUUnit('gru', size=D * 3)
        dy_ret = gru(
          base.to_variable(input), base.to_variable(hidden_input))




