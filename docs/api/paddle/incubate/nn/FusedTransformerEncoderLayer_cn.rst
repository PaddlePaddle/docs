.. _cn_api_paddle_incubate_nn_FusedTransformerEncoderLayer:

FusedTransformerEncoderLayer
-------------------------------
.. py:class:: paddle.incubate.nn.FusedTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout_rate=0.1, activation='relu', attn_dropout_rate=None, act_dropout_rate=None, normalize_before=False, weight_attr=None, bias_attr=None)


FusedTransformer 编码器层由两个子层组成：多头自注意力机制和前馈神经网络。如果 ``normalize_before`` 为 ``True``，则对每个子层的输入进行层标准化（Layer Normalization），对每个子层的输出进行 dropout 和残差连接（residual connection）。否则（即 ``normalize_before`` 为 ``False`` ），则对每个子层的输入不进行处理，只对每个子层的输出进行 dropout、残差连接（residual connection）和层标准化（Layer Normalization）。


参数
:::::::::
    - **d_model** (int) - 输入输出的维度。
    - **nhead** (int) - multi-head attention（MHA）的 Head 数量。
    - **dim_feedforward** (int) - 前馈神经网络中隐藏层的大小。
    - **dropout_rate** (float，可选) - 对两个子层的输出进行处理的 dropout 值，置零的概率。默认值：0.1。
    - **activation** (str，可选) - 前馈神经网络的激活函数。默认值：``relu``。
    - **attn_dropout_rate** (float，可选) - MHA 中对注意力目标的随机失活率。如果为 `None` 则 ``attn_dropout = dropout``。默认值：``None`` 。
    - **act_dropout_rate** (float，可选) - 前馈神经网络的激活函数后的 dropout 置零的概率。如果为 ``None`` 则 ``act_dropout_rate = dropout_rate``。默认值：``None`` 。
    - **normalize_before** (bool，可选) - 设置对每个子层的输入输出的处理。如果为 ``True``，则对每个子层的输入进行层标准化（Layer Normalization），否则（即为 ``False`` ），则对每个子层的输入不进行处理，而是在子层的输出前进行标准化。默认值：``False`` 。
    - **weight_attr** (ParamAttr|tuple，可选) - 指定权重参数属性的对象。如果是 ``tuple`` ，MHA 的权重参数属性使用 ``weight_attr[0]``，前馈神经网络的权重参数属性使用 ``weight_attr[1]``。如果参数值是 ``ParamAttr``，则 MHA 和前馈神经网络的权重参数属性都使用 ``ParamAttr``。默认值：``None``，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
    - **bias_attr** （ParamAttr|tuple|bool，可选）- 指定偏置参数属性的对象。如果是 ``tuple`` ，MHA 的偏置参数属性使用 ``bias_attr[0]``，前馈神经网络的偏置参数属性使用 ``bias_attr[1]``。如果该参数值是 ``ParamAttr``，则 MHA 和前馈神经网络的偏置参数属性都使用 ``ParamAttr``。如果该参数为 ``bool`` 类型，只支持为 ``False``，表示没有偏置参数。默认值为 None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。


返回
:::::::::
    - Tensor，输出 Tensor，数据类型与 ``x`` 一样。

代码示例
::::::::::

COPY-FROM: paddle.incubate.nn.FusedTransformerEncoderLayer
