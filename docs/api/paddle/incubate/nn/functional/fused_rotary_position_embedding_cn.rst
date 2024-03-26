.. _cn_api_paddle_incubate_nn_functional_fused_rotary_position_embedding:

fused_rotary_position_embedding
-------------------------------

.. py:function:: paddle.incubate.nn.functional.fused_rotary_position_embedding(q, k=None, v=None, sin=None, cos=None, position_ids=None, use_neox_rotary_style=True)
融合旋转位置编码。

参数
::::::::::

    - **q** (Tensor) - 输入张量。 数据类型可以是 bfloat16, float16, float32 或 float64。 q 的形状必须是 [batch_size, seq_len, num_heads, head_dim]，并且 head_dim 必须是 2 的倍数。
    - **k** (Tensor, 可选) - 输入张量。 数据类型可以是 bfloat16, float16, float32 或 float64。 k 的形状必须是 [batch_size, seq_len, num_heads, head_dim]，并且 head_dim 必须是 2 的倍数。
    - **v** (Tensor, 可选) - 输入张量。 数据类型可以是 bfloat16, float16, float32 或 float64。 v  的形状必须是 [batch_size, seq_len, num_heads, head_dim]，并且 head_dim 必须是 2 的倍数。
    - **sin** (Tensor, 可选) - 输入张量。 数据类型可以是 bfloat16, float16, float32 或 float64。 sin 的形状必须是 [seq_len, head_dim] 或 [1, seq_len, 1, head_dim], 并且 head_dim 必须是 2 的倍数。
    - **cos** (Tensor, 可选) - 输入张量。 数据类型可以是 bfloat16, float16, float32 或 float64。 cos 的形状必须是 [seq_len, head_dim] 或 [1, seq_len, 1, head_dim], 并且 head_dim 必须是 2 的倍数。
    - **position_ids** (Tensor, 可选) - 输入张量。 数据类型为 int64. position_ids 的形状为[batch_size, seq_len]。
    - **use_neox_rotary_style** (可选|bool) - 当 use_neox_rotary_style 为 True, 每两个相邻的数字计算一次。 当 use_neox_rotary_style 为 False, 计算与前半段和后半段位置相对应的数字。 默认值为 True。


返回
::::::::::

    - out_q/out_k/out_v 表示融合旋转位置嵌入的张量，具有与 `q` 相同的形状和数据类型。


代码示例
::::::::::

COPY-FROM: paddle.incubate.nn.functional.fused_rotary_position_embedding
