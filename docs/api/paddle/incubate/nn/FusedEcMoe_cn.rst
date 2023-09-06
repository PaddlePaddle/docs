.. _cn_api_incubate_nn_FusedEcMoe:

FusedEcMoE
-------------------------------
.. py:class:: paddle.incubate.nn.FusedEcMoe(hidden_size, inter_size, num_experts, act_type, weight_attr=None, bias_attr=None)

这是一个调用融合算子 fused_ec_moe（参考 :ref:`cn_api_incubate_nn_functional_fused_ec_moe` ）。


参数
:::::::::
    - **hidden_size** (int) - 输入输出的维度。
    - **inter_size** (int) - 前馈神经网络中隐藏层的大小。
    - **num_expert** (int) - 专家网络的数量。
    - **act_type** (string) - 激活函数类型，目前仅支持 ``gelu`` , ``relu``。
    - **weight_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值：``None``，表示使用默认的权重参数属性，即使用 0 进行初始化。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **bias_attr** （ParamAttr|bool，可选）- 指定偏置参数属性的对象。如果该参数值是 ``ParamAttr``，则使用 ``ParamAttr``。如果该参数为 ``bool`` 类型，只支持为 ``False``，表示没有偏置参数。默认值为 None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。

返回
:::::::::
    - Tensor，输出 Tensor，数据类型与 ``x`` 一样。

代码示例
::::::::::

COPY-FROM: paddle.incubate.nn.FusedEcMoe
