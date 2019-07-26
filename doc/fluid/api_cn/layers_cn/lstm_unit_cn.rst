.. _cn_api_fluid_layers_lstm_unit:

lstm_unit
-------------------------------

.. py:function:: paddle.fluid.layers.lstm_unit(x_t, hidden_t_prev, cell_t_prev, forget_bias=0.0, param_attr=None, bias_attr=None, name=None)

Lstm unit layer

lstm步的等式：

.. math::

    i_{t} &= \sigma \left ( W_{x_{i}}x_{t}+W_{h_{i}}h_{t-1}+b_{i} \right ) \\
    f_{t} &= \sigma \left ( W_{x_{f}}x_{t}+W_{h_{f}}h_{t-1}+b_{f} \right ) \\
    c_{t} &= f_{t}c_{t-1}+i_{t}tanh\left ( W_{x_{c}}x_{t} +W_{h_{c}}h_{t-1}+b_{c}\right ) \\
    o_{t} &= \sigma \left ( W_{x_{o}}x_{t}+W_{h_{o}}h_{t-1}+b_{o} \right ) \\
    h_{t} &= o_{t}tanh \left ( c_{t} \right )

lstm单元的输入包括 :math:`x_{t}` ， :math:`h_{t-1}` 和 :math:`c_{t-1}` 。:math:`h_{t-1}` 和 :math:`c_{t-1}` 的第二维应当相同。在此实现过程中，线性转换和非线性转换分离。以 :math:`i_{t}` 为例。线性转换运用到fc层，等式为：

.. math::

    L_{i_{t}} = W_{x_{i}}x_{t} + W_{h_{i}}h_{t-1} + b_{i}

非线性转换运用到lstm_unit运算，方程如下：

.. math::

    i_{t} = \sigma \left ( L_{i_{t}} \right )

该层有 :math:`h_{t}` 和 :math:`c_{t}` 两个输出。

参数：
    - **x_t** (Variable) - 当前步的输入值，二维张量，shape为 M x N ，M是批尺寸，N是输入尺寸
    - **hidden_t_prev** (Variable) - lstm单元的隐藏状态值，二维张量，shape为 M x S，M是批尺寸，N是lstm单元的大小
    - **cell_t_prev** (Variable) - lstm单元的cell值，二维张量，shape为 M x S ，M是批尺寸，N是lstm单元的大小
    - **forget_bias** (Variable) - lstm单元的遗忘bias
    - **param_attr** (ParamAttr|None) - 可学习hidden-hidden权重的擦参数属性。如果设为None或者ParamAttr的一个属性，lstm_unit创建ParamAttr为param_attr。如果param_attr的初始化函数未设置，参数初始化为Xavier。默认：None
    - **bias_attr** (ParamAttr|None) - 可学习bias权重的bias属性。如果设为False，输出单元中则不添加bias。如果设为None或者ParamAttr的一个属性，lstm_unit创建ParamAttr为bias_attr。如果bias_attr的初始化函数未设置，bias初始化为0.默认：None
    - **name** (str|None) - 该层名称（可选）。若设为None，则自动为该层命名

返回：lstm单元的hidden(隐藏状态)值和cell值

返回类型：tuple（元组）

抛出异常:
  - ``ValueError`` - ``x_t``，``hidden_t_prev`` 和 ``cell_t_prev`` 的阶不为2，或者 ``x_t`` ，``hidden_t_prev`` 和 ``cell_t_prev`` 的第一维不一致，或者 ``hidden_t_prev`` 和 ``cell_t_prev`` 的第二维不一致

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
     
    dict_dim, emb_dim, hidden_dim = 128, 64, 512
    data = fluid.layers.data(name='step_data', shape=[1], dtype='int32')
    x = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])
    pre_hidden = fluid.layers.data(name='pre_hidden', shape=[hidden_dim], dtype='float32')
    pre_cell = fluid.layers.data(name='pre_cell', shape=[hidden_dim], dtype='float32')
    hidden = fluid.layers.lstm_unit(
        x_t=x,
        hidden_t_prev=prev_hidden,
        cell_t_prev=prev_cell)











