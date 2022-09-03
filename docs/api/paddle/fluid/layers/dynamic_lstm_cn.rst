.. _cn_api_fluid_layers_dynamic_lstm:

dynamic_lstm
-------------------------------


.. py:function::  paddle.fluid.layers.dynamic_lstm(input, size, h_0=None, c_0=None, param_attr=None, bias_attr=None, use_peepholes=True, is_reverse=False, gate_activation='sigmoid', cell_activation='tanh', candidate_activation='tanh', dtype='float32', name=None)




该OP实现了 LSTM，即 Long-Short Term Memory（长短期记忆）运算 - `Hochreiter, S., & Schmidhuber, J. (1997) <https://www.scirp.org/pdf/JMF_2018013014134167.pdf>`_ 。

.. note::
    - 该OP仅支持 LoDTensor 作为输入，如果您需要处理的是Tensor，请使用 :ref:`cn_api_fluid_layers_lstm` 。
    - 在实现的时候为了提升效率，用户必须将LSTM的输入先进行线性映射，将维度为 [T, hidden_size] 的输入映射为 [T, 4 × hidden_size] 输入，然后再传给该OP。

该OP的默认实现方式为 diagonal/peephole 连接，参见 `Gers, F. A., & Schmidhuber, J. (2000) <ftp://ftp.idsia.ch/pub/juergen/TimeCount-IJCNN2000.pdf>`_ 。
如果需要禁用 peephole 连接方法，将 use_peepholes 设为 False 即可。

该OP对于序列中每一个时间步的计算公式如下：

.. math::
      i_t=\sigma (W_{ix}x_{t}+W_{ih}h_{t-1}+W_{ic}c_{t-1}+b_i)
.. math::
      f_t=\sigma (W_{fx}x_{t}+W_{fh}h_{t-1}+W_{fc}c_{t-1}+b_f)
.. math::
      o_t=\sigma (W_{ox}x_{t}+W_{oh}h_{t-1}+W_{oc}c_{t-1}+b_o)
.. math::
      \widetilde{c_t}=act_g(W_{ct}x_{t}+W_{ch}h_{t-1}+b_{c})
.. math::
      c_t=f_t\odot c_{t-1}+i_t\odot \widetilde{c_t}
.. math::
      h_t=o_t\odot act_h(c_t)

公式中的概念信息如下：
      - :math:`x_{t}` 表示时间步 :math:`t` 的输入
      - :math:`h_{t}` 表示时间步 :math:`t` 的 hidden 状态
      - :math:`h_{t-1}, c_{t-1}` 分别表示前一个时间步的 hidden 和 cell 状态
      - :math:`\widetilde{c_t}` 表示候选的 cell 状态
      - :math:`i_t` ，:math:`f_t` 和 :math:`o_t` 分别为 input gate，forget gate，output gate
      - :math:`W` 表示 weight （例如，:math:`W_{ix}` 是在计算 input gate :math:`i_t` 时，对输入 :math:`x_{t}` 做线性变换的 weight）
      - :math:`b` 表示 bias （例如，:math:`b_{i}` 是 input gate 的 bias）
      - :math:`\sigma` 表示 gate 的非线性激活函数，默认为 sigmoid
      - :math:`act_g， act_h` 分别表示 cell 输入和 cell 输出的非线性激活函数，默认为 tanh
      - :math:`\odot` 表示矩阵的 Hadamard product，即对两个维度相同的矩阵，将相同位置的元素相乘，得到另一个维度相同的矩阵

参数
::::::::::::

  - **input** ( :ref:`api_guide_Variable` ) 维度为 :math:`[T, 4*hidden\_size]` 的多维 LoDTensor（必须在传入该OP前对维度为 :math:`[T, hidden\_size]` 的输入经过线性变换得到），其中 T 为 batch 中所有样本的长度之和，hidden_size 为隐层大小，数据类型为 float32 或者 float64。
  - **size** (int) – 必须为 4*hidden_size。
  - **h_0** ( :ref:`api_guide_Variable`，可选) 维度为 :math:`[batch\_size, hidden\_size]` 的多维 Tensor，其中 hidden_size 为隐层大小，数据类型为 float32 或者 float64。如果为 None，该OP会自动设置为全0的向量。默认值为None。
  - **c_0** ( :ref:`api_guide_Variable`，可选) 维度为 :math:`[batch\_size, hidden\_size]` 的多维 Tensor，其中 hidden_size 为隐层大小，数据类型为 float32 或者 float64。如果为 None，该OP会自动设置为全0的向量；:math:`h_0, c_0` 如果要设置为None，必须同时为None。默认值为None。
  - **param_attr** (ParamAttr，可选) – 指定权重参数属性的对象。如果为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr`。如果用户需要设置此属性，维度必须等于 :math:`[hidden\_size, 4*hidden\_size]`。默认值为None。
  - **bias_attr** (ParamAttr，可选) – 指定偏置参数属性的对象。如果为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr`。如果用户需要设置此属性，如果 use_peepholes=true，维度需为 :math:`[1, 4*hidden\_size]`, use_peepholes=true，维度需为 :math:`[1, 7*hidden\_size]`。默认值为None。   
  - **use_peepholes** (bool，可选) – 是否使用 peephole 连接。默认值为True。
  - **is_reverse** (bool，可选) – 是否将输入的数据根据根据样本长度进行逆序，同时会将输出进行逆序，用户拿到结果之后，不需要再逆序。默认值为False。
  - **gate_activation** (str，可选) – 应用于input gate，forget gate， output gate 的激活函数。默认值为sigmoid。
  - **cell_activation** (str，可选) – 用于cell输入的激活函数。默认值为tanh。
  - **candidate_activation** (str，可选) – 用于cell输出的激活函数。默认值为tanh。
  - **dtype** (str，可选) – 数据类型为 float32 或者 float64。默认值为 float32。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
经过lstm运算输出的 hidden 和 cell 的状态的tuple，包括

- hidden：LSTM hidden的输出结果，维度为 :math:`[T, hidden\_size]` 的LoDTensor，且LoD保持与输入一致，数据类型与input一致。
- cell：LSTM cell的输出结果，维度为 :math:`[T, hidden\_size]` 的LoDTensor，且LoD保持与输入一致，数据类型与input一致。

返回类型
::::::::::::
 tuple（ :ref:`api_guide_Variable` , :ref:`api_guide_Variable` ）


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.dynamic_lstm