.. _cn_api_fluid_layers_lstm_unit:

lstm_unit
-------------------------------


.. py:function:: paddle.fluid.layers.lstm_unit(x_t, hidden_t_prev, cell_t_prev, forget_bias=0.0, param_attr=None, bias_attr=None, name=None)





Long-Short Term Memory（LSTM）循环神经网络计算单元。该OP用于完成单个时间步内LSTM的计算，基于论文 `RECURRENT NEURAL NETWORK REGULARIZATION <http://arxiv.org/abs/1409.2329>`_ 中的描述实现，

并在forget gate（遗忘门）中增加了 ``forget_bias`` 来控制遗忘力度，公式如下：

.. math::

    i_{t} &= \sigma \left ( W_{x_{i}}x_{t}+W_{h_{i}}h_{t-1}+b_{i} \right ) \\
    f_{t} &= \sigma \left ( W_{x_{f}}x_{t}+W_{h_{f}}h_{t-1}+b_{f}+forget\_bias \right ) \\
    c_{t} &= f_{t}c_{t-1}+i_{t}tanh\left ( W_{x_{c}}x_{t} +W_{h_{c}}h_{t-1}+b_{c}\right ) \\
    o_{t} &= \sigma \left ( W_{x_{o}}x_{t}+W_{h_{o}}h_{t-1}+b_{o} \right ) \\
    h_{t} &= o_{t}tanh \left ( c_{t} \right )

其中，:math:`x_{t}` 对应 ``x_t``，表示当前时间步的输入；:math:`h_{t-1}` 和 :math:`c_{t-1}` 对应 ``hidden_t_prev`` 和 ``cell_t_prev``，表示上一时间步的hidden和cell输出；
:math:`i_{t}, f_{t}, c_{t}, o_{t}, h_{t}` 分别为input gate（输入门）、forget gate（遗忘门）、cell、output gate（输出门）和hidden的计算。


参数
::::::::::::

    - **x_t** (Variable) - 表示当前时间步的输入的Tensor，形状为 :math:`[N, M]`，其中 :math:`N` 为batch_size， :math:`M` 为输入的特征维度大小。数据类型为float32或float64。
    - **hidden_t_prev** (Variable) - 表示前一时间步hidden输出的Tensor，形状为 :math:`[N, D]`，其中 :math:`N` 为batch_size， :math:`D` 为LSTM中隐单元的数目。数据类型与 ``x_t`` 相同。
    - **cell_t_prev** (Variable) - 表示前一时间步cell输出的Tensor，和  ``hidden_t_prev`` 具有相同形状和数据类型。
    - **forget_bias** (float，可选) - 额外添加在遗忘门中的偏置项(参见公式)。默认值为0。
    - **param_attr** (ParamAttr，可选) – 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **bias_attr** (ParamAttr，可选) - 指定偏置参数属性的对象。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **name**  (str，可选) – 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为None。

返回
::::::::::::
Variable的二元组，包含了两个形状和数据类型均与 ``hidden_t_prev`` 相同的Tensor，分别表示hiddel和cell输出，即公式中的 :math:`h_{t}` 和 :math:`c_{t}` 。

返回类型
::::::::::::
tuple

抛出异常
::::::::::::

    - :code:`ValueError`： ``x_t`` 的阶不为2
    - :code:`ValueError`： ``hidden_t_prev`` 的阶不为2
    - :code:`ValueError`： ``cell_t_prev`` 的阶不为2
    - :code:`ValueError`： ``x_t`` 、``hidden_t_prev`` 和 ``cell_t_prev`` 的第一维大小必须相同
    - :code:`ValueError`： ``hidden_t_prev`` 和 ``cell_t_prev`` 的第二维大小必须相同


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.lstm_unit