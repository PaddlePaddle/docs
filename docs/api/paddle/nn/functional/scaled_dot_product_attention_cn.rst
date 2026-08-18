.. _cn_api_paddle_nn_functional_scaled_dot_product_attention:

scaled_dot_product_attention
-------------------------------

.. py:function:: paddle.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, training=True, backend=None, scale=None, enable_gqa=True, name=None)
计算公式为:

..  math::
    result=softmax(\frac{ Q * K^T }{\sqrt{d}}) * V
其中, ``Q``、``K`` 和 ``V`` 表示注意力模块的三个输入参数。这三个参数的尺寸相同。``d`` 表示三个参数中最后一个维度的大小。

.. warning::
    此 API 仅对数据类型为 float16 和 bfloat16 的输入进行校验；其他数据类型可能回退到优化程度较低的 math 实现。

.. warning::
    如果 ``is_causal`` 为 True，不应同时提供因果掩码；否则提供的掩码会被忽略。

.. note::
    本 API 的 QKV 布局为 ``[batch_size, seq_len, num_heads, head_dim]`` 或 ``[seq_len, num_heads, head_dim]``。若需要 ``num_heads`` 位于 ``seq_len`` 之前的布局，请使用 ``paddle.compat.nn.functional.scaled_dot_product_attention``。


参数
::::::::::

    - **query** (Tensor) - 注意力模块中的查询张量。具有以下形状的四维张量：[batch_size, seq_len, num_heads, head_dim]，或者三维张量：[seq_len, num_heads, head_dim]。数据类型可以是 float16 或 bfloat16。
    - **key** (Tensor) - 注意力模块中的关键张量。具有以下形状的四维张量:[batch_size, seq_len, num_heads, head_dim]，或者三维张量：[seq_len, num_heads, head_dim]。数据类型可以是 float16 或 bfloat16。
    - **value** (Tensor) - 注意力模块中的值张量。具有以下形状的四维张量: [batch_size, seq_len, num_heads, head_dim]，或者三维张量：[seq_len, num_heads, head_dim]。数据类型可以是 float16 或 bfloat16。
    - **attn_mask** (Tensor, 可选) - 注意力掩码张量，形状应可广播到 ``[batch_size, num_heads, seq_len_key, seq_len_query]``。数据类型可以是 bool 或与 ``query`` 相同的数据类型。bool 掩码中的 True 表示该位置参与注意力计算；非 bool 掩码会被加到注意力分数上。默认值为 None。
    - **dropout_p** (float) - ``dropout`` 的比例, 默认值为 0.00 即不进行正则化。
    - **is_causal** (bool) - 是否启用因果关系, 默认值为 False 即不启用。
    - **training** (bool): - 是否处于训练阶段, 默认值为 True 即处于训练阶段。
    - **backend** (str，可选) - 指定计算 scaled dot product attention 的后端。目前仅支持用于分布式场景的 ``"p2p"``。默认值为 None。
    - **scale** (float，可选) - 计算注意力权重时使用的缩放因子。为 None 时，使用 ``1 / sqrt(head_dim)``。默认值为 None。
    - **enable_gqa** (bool，可选) - 是否启用 GQA（Group Query Attention）模式。默认值为 True。
    - **name** (str, 可选) - 默认值为 None。通常不需要用户设置此属性。欲了解更多信息, 请参阅:ref:`api_guide_Name`。


返回
::::::::::

    - ``out`` (Tensor): 形状为 ``[batch_size, seq_len, num_heads, head_dim]`` 的 4 维张量或者形状为 ``[seq_len, num_heads, head_dim]`` 的 3 维张量。数据类型可以是 float16 或 bfloat16。


代码示例
::::::::::

COPY-FROM: paddle.nn.functional.scaled_dot_product_attention
