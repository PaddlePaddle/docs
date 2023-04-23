.. _cn_api_fluid_layers_LSTMCell:

LSTMCell
-------------------------------



.. py:class:: paddle.fluid.layers.LSTMCell(hidden_size, param_attr=None, bias_attr=None, gate_activation=None, activation=None, forget_bias=1.0, dtype="float32", name="LSTMCell")



    
长短期记忆单元（Long-Short Term Memory）。通过对 :code:`fluid.contrib.layers.rnn_impl.BasicLSTMUnit` 包装，来让它可以应用于RNNCell。    

公式如下：
  
.. math:: 
    i_{t} &= act_g \left ( W_{x_{i}}x_{t}+W_{h_{i}}h_{t-1}+b_{i} \right ) \\
    f_{t} &= act_g \left ( W_{x_{f}}x_{t}+W_{h_{f}}h_{t-1}+b_{f}+forget\_bias \right ) \\
    c_{t} &= f_{t}c_{t-1}+i_{t}act_h\left ( W_{x_{c}}x_{t} +W_{h_{c}}h_{t-1}+b_{c}\right ) \\
    o_{t} &= act_g\left ( W_{x_{o}}x_{t}+W_{h_{o}}h_{t-1}+b_{o} \right ) \\
    h_{t} &= o_{t}act_h \left ( c_{t} \right )

更多细节可以参考 `RECURRENT NEURAL NETWORK REGULARIZATION <http://arxiv.org/abs/1409.2329>`_ 

参数
::::::::::::

  - **hidden_size** (int) - LSTMCell中的隐藏层大小。
  - **param_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr`。
  - **bias_attr** (ParamAttr，可选) - 指定偏置参数属性的对象。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。 
  - **gate_activation** (function，可选) - :math:`act_g` 的激活函数。默认值为 :code:`fluid.layers.sigmoid`。 
  - **activation** (function，可选) - :math:`act_c` 的激活函数。默认值为 :code:`fluid.layers.tanh`。
  - **forget_bias** (float，可选) - 计算遗忘们时使用的遗忘偏置。默认值为 1.0。
  - **dtype** (string，可选) - 此Cell中使用的数据类型。默认值为 `float32`。 
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
LSTMCell类的实例对象。

代码示例
::::::::::::


COPY-FROM: paddle.fluid.layers.LSTMCell

方法
::::::::::::
call(inputs, states)
'''''''''

执行GRU的计算。
    
**参数**

  - **input** (Variable) - 输入，形状为 :math:`[batch\_size，input\_size]` 的tensor，对应于公式中的 :math:`x_t`。数据类型应为float32。 
  - **states** (Variable) - 状态，包含两个tensor的列表，每个tensor形状为 :math:`[batch\_size，hidden\_size]`。对应于公式中的 :math:`h_{t-1}, c_{t-1}`。数据类型应为float32。 
    
**返回**
一个元组 :code:`(outputs, new_states)`，其中 :code:`outputs` 是形状为 :math:`[batch\_size，hidden\_size]` 的tensor，对应于公式中的 :math:`h_{t}`；:code:`new_states` 是一个列表，包含形状为 :math:`[batch_size，hidden_size]` 的两个tensor变量，它们对应于公式中的 :math:`h_{t}, c_{t}`。这些tensor的数据类型都与 :code:`state` 的数据类型相同。

**返回类型**
tuple

state_shape()
'''''''''

LSTMCell的 :code:`state_shape` 是一个具有两个形状的列表：:math:`[[hidden\_size], [hidden\_size]]` （batch大小为-1，自动插入到形状中）。这两个形状分别对应于公式中的 :math:`h_{t-1}` and :math:`c_{t-1}`。

**参数**
无。

**返回**
LSTMCell的 :code:`state_shape` 

**返回类型**
list
