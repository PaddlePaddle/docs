.. _cn_api_paddle_incubate_nn_functional_fused_ec_moe:

fused_ec_moe
-------------------------------

.. py:function:: paddle.incubate.nn.functional.fused_ec_moe(x, gate, bmm0_weight, bmm0_bias, bmm1_weight, bmm1_bias, act_type)

该算子实现了 EcMoE 的融合版本，目前只支持在 sm75，sm80，sm86 架构下的 GPU 上使用。

参数
:::::::::
    - **x** (Tensor) - 输入 Tensor，形状是 ``[bsz, seq_len, d_model]`` 。
    - **gate** (Tensor) - 用于选择专家的 gate Tensor，形状是 ``[bsz, seq_len, e]`` 。
    - **bmm0_weight** (Tensor) - 第一个 batch matmul 的权重数据，形状是 ``[e, d_model, d_feed_forward]``。
    - **bmm0_bias** (Tensor) - 第一个 batch matmul 的偏置数据，形状是 ``[e, 1, d_feed_forward]``。
    - **bmm1_weight** (Tensor) - 第二个 batch matmul 的权重数据，形状是 ``[e, d_model, d_feed_forward]``。
    - **bmm1_bias** (Tensor) - 第二个 batch matmul 的偏置数据，形状是 ``[e, 1, d_feed_forward]``。
    - **act_type** (string) - 激活函数类型，目前仅支持 ``gelu`` , ``relu``。

返回
:::::::::
    - Tensor，输出 Tensor，数据类型与 ``x`` 一样。

代码示例
::::::::::

COPY-FROM: paddle.incubate.nn.functional.fused_ec_moe
