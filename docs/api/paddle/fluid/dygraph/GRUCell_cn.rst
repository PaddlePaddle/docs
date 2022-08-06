.. _cn_api_fluid_layers_GRUCell:

GRUCell
-------------------------------


.. py:class:: paddle.fluid.layers.GRUCell(hidden_size, param_attr=None, bias_attr=None, gate_activation=None, activation=None, dtype="float32", name="GRUCell")



    
门控循环单元（Gated Recurrent Unit）。通过对 :code:`fluid.contrib.layers.rnn_impl.BasicGRUUnit` 包装，来让它可以应用于RNNCell。

公式如下：

.. math::
    u_t & = act_g(W_{ux}x_{t} + W_{uh}h_{t-1} + b_u)\\
    r_t & = act_g(W_{rx}x_{t} + W_{rh}h_{t-1} + b_r)\\
    \tilde{h_t} & = act_c(W_{cx}x_{t} + W_{ch}(r_t \odot h_{t-1}) + b_c)\\
    h_t & = u_t \odot h_{t-1} + (1-u_t) \odot \tilde{h_t}

更多细节可以参考 `Learning Phrase Representations using RNN Encoder Decoder for Statistical Machine Translation <https://arxiv.org/pdf/1406.1078.pdf>`_ 
  
参数
::::::::::::

  - **hidden_size** (int) - GRUCell中的隐藏层大小。
  - **param_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr`。
  - **bias_attr** (ParamAttr，可选) - 指定偏置参数属性的对象。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。 
  - **gate_activation** (function，可选) - :math:`act_g` 的激活函数。默认值为 :code:`fluid.layers.sigmoid`。 
  - **activation** (function，可选) - :math:`act_c` 的激活函数。默认值为 :code:`fluid.layers.tanh` 
  - **dtype** (string，可选) - 此cell中使用的数据类型。默认为"float32"。 
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
GRUCell类的实例对象。

代码示例
::::::::::::


COPY-FROM: paddle.fluid.layers.GRUCell

方法
::::::::::::
call(inputs, states)
'''''''''

执行GRU的计算。
    
**参数**

  - **input** (Variable) - 输入，形状为 :math:`[batch\_size，input\_size]` 的tensor，对应于公式中的 :math:`x_t`。数据类型应为float32。 
  - **states** (Variable) - 状态，形状为 :math:`[batch\_size，hidden\_size]` 的tensor。对应于公式中的 :math:`h_{t-1}`。数据类型应为float32。 
    
**返回**
一个元组 :code:`(outputs, new_states)`，其中 :code:`outputs` 和 :code:`new_states` 是同一个tensor，其形状为 :math:`[batch\_size，hidden\_size]`，数据类型和 :code:`state` 的数据类型相同，对应于公式中的 :math:`h_t`。

**返回类型**
tuple

state_shape()
'''''''''

GRUCell的 :code:`state_shape` 是形状 :math:`[hidden\_size]` （batch大小为-1，自动插入到形状中），对应于 :math:`h_{t-1}` 的形状。

**参数**
无。

**返回**
GRUCell的 :code:`state_shape`。

**返回类型**
Variable


