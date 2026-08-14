.. _cn_api_paddle_incubate_nn_FusedFeedForward:

FusedFeedForward
-------------------------------
.. py:class:: paddle.incubate.nn.FusedFeedForward(d_model, dim_feedforward, dropout_rate=0.1, epsilon=1e-05, activation='relu', act_dropout_rate=None, normalize_before=False, linear1_weight_attr=None, linear1_bias_attr=None, linear2_weight_attr=None, linear2_bias_attr=None, ln1_scale_attr=None, ln1_bias_attr=None, ln2_scale_attr=None, ln2_bias_attr=None, nranks=1, ring_id=-1, name=None)

这是一个调用融合算子 fused_feedforward（参考 :ref:`cn_api_paddle_incubate_nn_functional_fused_feedforward` ）。


参数
:::::::::
    - **d_model** (int) - 输入输出的维度。
    - **dim_feedforward** (int) - 前馈神经网络中隐藏层的大小。
    - **dropout_rate** (float，可选) - 对本层的输出进行处理的 dropout 值，置零的概率。默认值：0.1。
    - **epsilon** (float，可选) - 为防止方差除零而加到方差上的小值。默认值为 1e-05。
    - **activation** (str，可选) - 激活函数。默认值：``relu``。
    - **act_dropout_rate** (float，可选) - 激活函数后的 dropout 置零的概率。如果为 ``None`` 则  ``act_dropout_rate = dropout_rate``。默认值：``None`` 。
    - **normalize_before** (bool，可选) - 设置对输入输出的处理。如果为 ``True``，则对输入进行层标准化（Layer Normalization），否则（即为 ``False`` ），则对输入不进行处理，而是在输出前进行标准化。默认值：``False`` 。
    - **linear1_weight_attr** (ParamAttr，可选) - 前馈网络第一个线性层权重参数的属性。默认值为 None。具体用法请参见 :ref:`cn_api_paddle_ParamAttr`。
    - **linear1_bias_attr** (ParamAttr|bool，可选) - 前馈网络第一个线性层偏置参数的属性。设为 False 时该层不包含可训练偏置参数。默认值为 None。具体用法请参见 :ref:`cn_api_paddle_ParamAttr`。
    - **linear2_weight_attr** (ParamAttr，可选) - 前馈网络第二个线性层权重参数的属性。默认值为 None。具体用法请参见 :ref:`cn_api_paddle_ParamAttr`。
    - **linear2_bias_attr** (ParamAttr|bool，可选) - 前馈网络第二个线性层偏置参数的属性。设为 False 时该层不包含可训练偏置参数。默认值为 None。具体用法请参见 :ref:`cn_api_paddle_ParamAttr`。
    - **ln1_scale_attr** (ParamAttr，可选) - 前置 LayerNorm 权重参数的属性。默认值为 None。具体用法请参见 :ref:`cn_api_paddle_ParamAttr`。
    - **ln1_bias_attr** (ParamAttr|bool，可选) - 前置 LayerNorm 偏置参数的属性。设为 False 时该层不包含可训练偏置参数。默认值为 None。具体用法请参见 :ref:`cn_api_paddle_ParamAttr`。
    - **ln2_scale_attr** (ParamAttr，可选) - 后置 LayerNorm 权重参数的属性。默认值为 None。具体用法请参见 :ref:`cn_api_paddle_ParamAttr`。
    - **ln2_bias_attr** (ParamAttr|bool，可选) - 后置 LayerNorm 偏置参数的属性。设为 False 时该层不包含可训练偏置参数。默认值为 None。具体用法请参见 :ref:`cn_api_paddle_ParamAttr`。
    - **nranks** (int，可选) - 分布式张量模型并行的 rank 数。默认值为 1，表示不使用张量并行。
    - **ring_id** (int，可选) - 用于分布式张量模型并行。默认值为 -1，表示不使用张量并行。
    - **name** (str|None，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
    - Tensor，输出 Tensor，数据类型与 ``x`` 一样。

代码示例
::::::::::

COPY-FROM: paddle.incubate.nn.FusedFeedForward
