.. _cn_api_paddle_compat_nn_functional_scaled_dot_product_attention:

scaled_dot_product_attention
-------------------------------

.. py:function:: paddle.compat.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False)

PyTorch 兼容的缩放点积注意力。与原生接口不同，本接口的 Q、K、V 布局为 ``[batch_size, num_heads, seq_len, head_dim]`` 或 ``[num_heads, seq_len, head_dim]``。

参数
:::::::::

    - **query** (Tensor) - 查询 Tensor。
    - **key** (Tensor) - 键 Tensor。
    - **value** (Tensor) - 值 Tensor。
    - **attn_mask** (Tensor|None，可选) - 可广播到注意力分数形状的掩码。默认值：None。
    - **dropout_p** (float，可选) - 注意力权重的 dropout 概率。默认值：0.0。
    - **is_causal** (bool，可选) - 是否使用因果掩码。为 True 时不能同时传入 ``attn_mask``。默认值：False。
    - **scale** (float|None，可选) - 注意力权重的缩放因子；None 表示使用 ``1 / sqrt(head_dim)``。默认值：None。
    - **enable_gqa** (bool，可选) - 是否启用 GQA。默认值：False。

返回
:::::::::

Tensor，缩放点积注意力的结果。

代码示例
:::::::::

COPY-FROM: paddle.compat.nn.functional.scaled_dot_product_attention
