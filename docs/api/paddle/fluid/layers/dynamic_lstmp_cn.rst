.. _cn_api_fluid_layers_dynamic_lstmp:

dynamic_lstmp
-------------------------------

.. py:function:: paddle.fluid.layers.dynamic_lstmp(input, size, proj_size, param_attr=None, bias_attr=None, use_peepholes=True, is_reverse=False, gate_activation='sigmoid', cell_activation='tanh', candidate_activation='tanh', proj_activation='tanh', dtype='float32', name=None, h_0=None, c_0=None, cell_clip=None, proj_clip=None)




.. note::
    在实现的时候为了提升效率，用户必须将输入先进行线性映射，将维度为 [T, hidden_size] 的输入映射为 [T, 4×hidden_size] 输入，然后再传给该OP。

该OP实现了LSTMP（LSTM Projected）层。LSTMP层在LSTM层之后有一个单独的的线性映射层。-- `Sak, H., Senior, A., & Beaufays, F. (2014) <https://ai.google/research/pubs/pub43905.pdf>`_ 。

与标准的LSTM层相比，LSTMP多出来的线性映射层，用于从原始隐藏状态 :math:`h_t` 映射到较低维的状态 :math:`r_t`，
从而减少参数总数和计算复杂度，特别是输出单元相对较大的情况下。

该OP的默认实现方式为 diagonal/peephole 连接，参见 `Gers, F. A., & Schmidhuber, J. (2000) <ftp://ftp.idsia.ch/pub/juergen/TimeCount-IJCNN2000.pdf>`_ 。
如果需要禁用 peephole 连接方法，将 use_peepholes 设为 False 即可。

该OP对于序列中每一个时间步的计算公式如下：

.. math::
      i_t = \sigma(W_{ix}x_{t} + W_{ir}r_{t-1} + W_{ic}c_{t-1} + b_i)
.. math::
      f_t = \sigma(W_{fx}x_{t} + W_{fr}r_{t-1} + W_{fc}c_{t-1} + b_f)
.. math::
      o_t = \sigma(W_{ox}x_{t} + W_{or}r_{t-1} + W_{oc}c_{t-1} + b_o)
.. math::
      \widetilde{c_t} = act_g(W_{cx}x_t + W_{cr}r_{t-1} + b_c)
.. math::
      c_t = f_t \odot c_{t-1} + i_t \odot \widetilde{c_t}
.. math::
      h_t = o_t \odot act_h(c_t)
.. math::
      r_t = \overline{act_h}(W_{rh}h_t)


公式中的概念信息如下：
      - :math:`x_{t}` 表示时间步 :math:`t` 的输入
      - :math:`h_{t}` 表示时间步 :math:`t` 的 hidden 状态
      - :math:`r_{t}`：隐藏状态循环的映射输出的状态
      - :math:`h_{t-1}, c_{t-1}, r_{t-1}` 分别表示前一个时间步的 hidden 状态，cell 状态和循环映射输出状态
      - :math:`\widetilde{c_t}` 表示候选的 cell 状态
      - :math:`i_t` ，:math:`f_t` 和 :math:`o_t` 分别为 input gate，forget gate，output gate
      - :math:`W` 表示 weight （例如，:math:`W_{ix}` 是在计算 input gate :math:`i_t` 时，对输入 :math:`x_{t}` 做线性变换的 weight）
      - :math:`b` 表示 bias （例如，:math:`b_{i}` 是 input gate 的 bias）
      - :math:`\sigma` 表示 gate 的非线性激活函数，默认为 sigmoid
      - :math:`act_g, act_h, \overline{act_h}` 分别表示 cell 输入 cell 输出和映射输出的非线性激活函数，默认为 tanh
      - :math:`\odot` 表示矩阵的 Hadamard product，即对两个维度相同的矩阵，将相同位置的元素相乘，得到另一个维度相同的矩阵

参数
::::::::::::

  - **input** ( :ref:`api_guide_Variable` ) 维度为 :math:`[T, 4*hidden\_size]` 的多维 LoDTensor（必须在传入该OP前对维度为 :math:`[T, hidden\_size]` 的输入经过线性变换得到），其中 T 为 batch 中所有样本的长度之和，hidden_size 为隐层大小，数据类型为 float32 或者 float64。
  - **size** (int) – 必须为 4 * hidden_size。
  - **proj_size** (int) - 投影映射输出的大小。
  - **param_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。

    说明：
      1. 隐藏状态到隐藏状态（Hidden-hidden）权重 = :math:`\{ W_{cr},W_{ir},W_{fr},W_{or} \}`，维度为 [P, 4*hidden_size] ，P是投影大小
      
      2. 投影（Projection）权重 = :math:`\{ W_{rh} \}`，维度为 [D, P]

  - **bias_attr** (ParamAttr，可选) - 指定偏置参数属性的对象。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。

    说明：
      1. use_peepholes = False
          - Biases = { :math:`b_{c},b_{i},b_{f},b_{o}`}
          - 维度为 [1, 4*hidden_size]

      2. use_peepholes = True
          - Biases = { :math:`b_{c},b_{i},b_{f},b_{o},W_{ic},W_{fc},W_{oc}`}
          - 维度为 [1, 7*hidden_size]

  - **use_peepholes** (bool，可选) - 是否使用 peephole 连接。默认值为True。
  - **is_reverse** (bool，可选) - 是否计算反向LSTM，默认值为False。
  - **gate_activation** (str，可选) - 应用于input gate，forget gate， output gate 的激活函数。可选值包括 sigmoid，tanh，relu，identity。默认值为 sigmoid。
  - **cell_activation** (str，可选) - cell输出的激活函数。可选值包括 sigmoid，tanh，relu，identity。默认值为 tanh。
  - **candidate_activation** (str，可选) - 候选隐藏状态（candidate hidden state）的激活状态。可选值包括 sigmoid，tanh，relu，identity。默认值为 tanh。
  - **proj_activation** (str，可选) - 投影输出的激活函数。可选值包括 sigmoid，tanh，relu，identity。默认值为 tanh。
  - **dtype** (str，可选) - 数据类型。可选值包括 float32，float64。默认值为 float32。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
  - **h_0** ( :ref:`api_guide_Variable`，可选) 维度为 :math:`[batch\_size, hidden\_size]` 的多维 Tensor。如果为 None，该OP会自动设置为全0的向量。默认值为None。
  - **c_0** ( :ref:`api_guide_Variable`，可选) 维度为 :math:`[batch\_size, hidden\_size]` 的多维 Tensor。如果为 None，该OP会自动设置为全0的向量；:math:`h_0, c_0` 如果要设置为None，必须同时为None。默认值为None。
  - **cell_clip** (float，可选) - 如果该参数不为None，则在单元输出激活之前，单元状态将被此值剪裁。默认值为None。
  - **proj_clip** (float，可选) - 如果 num_proj > 0 并且 proj_clip 不为None，那么将投影值沿元素方向剪切到[-proj_clip，proj_clip]内。默认值为None。

返回
::::::::::::
经过lstmp运算输出的 hidden 的映射和 cell 状态的tuple，包括

- hidden：LSTM hidden的输出结果，维度为 :math:`[T, P]` 的LoDTensor，且LoD保持与输入一致，数据类型与input一致。
- cell：LSTM cell的输出结果，维度为 :math:`[T, hidden\_size]` 的LoDTensor，且LoD保持与输入一致，数据类型与input一致。

返回类型
::::::::::::
 tuple（ :ref:`api_guide_Variable` , :ref:`api_guide_Variable` ）

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.dynamic_lstmp