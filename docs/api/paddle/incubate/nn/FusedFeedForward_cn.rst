.. _cn_api_incubate_nn_FusedFeedForward:

FusedFeedForward
-------------------------------
.. py:class:: paddle.incubate.nn.FusedFeedForward(d_model, dim_feedforward, dropout_rate=0.1, activation='relu', act_dropout_rate=None, normalize_before=False, weight_attr=None, bias_attr=None)

这是一个调用融合算子 fused_feedforward（参考 :ref:`cn_api_incubate_nn_functional_fused_feedforward` ）。


参数
:::::::::
    - **d_model** (int) - 输入输出的维度。
    - **dim_feedforward** (int) - 前馈神经网络中隐藏层的大小。
    - **dropout_rate** (float，可选) - 对本层的输出进行处理的 dropout 值，置零的概率。默认值：0.1。
    - **activation** (str，可选) - 激活函数。默认值：``relu``。
    - **act_dropout_rate** (float，可选) - 激活函数后的 dropout 置零的概率。如果为 ``None`` 则  ``act_dropout_rate = dropout_rate``。默认值：``None`` 。
    - **normalize_before** (bool，可选) - 设置对输入输出的处理。如果为 ``True``，则对输入进行层标准化（Layer Normalization），否则（即为 ``False`` ），则对输入不进行处理，而是在输出前进行标准化。默认值：``False`` 。
    - **weight_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值：``None``，表示使用默认的权重参数属性，即使用 0 进行初始化。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **bias_attr** （ParamAttr|bool，可选）- 指定偏置参数属性的对象。如果该参数值是 ``ParamAttr``，则使用 ``ParamAttr``。如果该参数为 ``bool`` 类型，只支持为 ``False``，表示没有偏置参数。默认值为 None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。

返回
:::::::::
    - Tensor，输出 Tensor，数据类型与 ``x`` 一样。

代码示例
::::::::::

COPY-FROM: paddle.incubate.nn.FusedFeedForward
