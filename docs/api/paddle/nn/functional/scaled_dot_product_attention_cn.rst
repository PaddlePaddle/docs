.. _cn_api_paddle_nn_functional_scaled_dot_product_attention:

scaled_dot_product_attention
-------------------------------

.. py:function:: paddle.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, training=True, name=None)

计算公式为:

..  math::
    result=softmax(\frac{ Q * K^T }{\sqrt{d}}) * V

其中, ``Q``、``K`` 和 ``V`` 表示注意力模块的三个输入参数。这三个参数的尺寸相同。``d`` 表示三个参数中最后一个维度的大小。

.. warning::
    此 API 仅支持数据类型为 float16 和 bfloat16 的输入。


参数
::::::::::

    - **query** (Tensor) - 注意力模块中的查询张量。具有以下形状的四维张量：[batch_size, seq_len, num_heads, head_dim]。数据类型可以是 float61 或 bfloat16。
    - **key** (Tensor) - 注意力模块中的关键张量。具有以下形状的四维张量:[batch_size, seq_len, num_heads, head_dim]。数据类型可以是 float61 或 bfloat16。
    - **value** (Tensor) - 注意力模块中的值张量。具有以下形状的四维张量: [batch_size, seq_len, num_heads, head_dim]。数据类型可以是 float61 或 bfloat16。
    - **attn_mask** (Tensor, 可选) - 与添加到注意力分数的 ``query``、 ``key``、 ``value`` 类型相同的浮点掩码, 默认值为空。
    - **dropout_p** (float) - ``dropout`` 的比例, 默认值为 0.00 即不进行正则化。
    - **is_causal** (bool) - 是否启用因果关系, 默认值为 False 即不启用。
    - **training** (bool): - 是否处于训练阶段, 默认值为 True 即处于训练阶段。
    - **name** (str, 可选) - 默认值为 None。通常不需要用户设置此属性。欲了解更多信息, 请参阅:ref:`api_guide_Name`。


返回
::::::::::

    - ``out`` (Tensor): 形状为 ``[batch_size, seq_len, num_heads, head_dim]`` 的 4 维张量。数据类型可以是 float16 或 bfloat16。
    - ``softmax`` (Tensor): 如果 return_softmax 为 False,则为 None。


代码示例
::::::::::

COPY-FROM: paddle.nn.functional.scaled_dot_product_attention
