.. _cn_api_paddle_nn_functional_flash_attention_with_sparse_mask:

flash_attention_with_sparse_mask
-------------------------------

.. py:function:: paddle.nn.functional.flash_attention_with_sparse_mask(query, key, value, attn_mask_start_row_indices, attn_mask_start_row=0, dropout_p=0.0, is_causal=False, return_softmax=False, return_softmax_lse=False, return_seed_offset=False, training=True, name=None)

用稀疏mask表达的flash_attention。

.. math::

    result = softmax(\frac{ Q * K^T }{\sqrt{d}} + mask) * V

参数
::::::::::::

    - **query** (int) - 输入Query Tensor，shape=[batch_size, seq_len, num_heads, head_dim]，数据类型为 float16, bfloat16。
    - **key** (Tensor) - 输入Key Tensor，shape以及dtype和query相同。
    - **value** (Tensor) - 输入Value Tensor，shape以及dtype和query相同。
    - **attn_mask_start_row_indices** (Tensor) - 稀疏掩码索引，shape=[batch_size, num_head, seq_len]，每个元素的值表示得分矩阵中掩码开始的行索引。数据类型必须是 int32。
    - **attn_mask_start_row** (Tensor，可选) - 当传入attn_mask_start_row_indices并且已知最小行数大于0时，可以设置attn_mask_start_row以提高性能。默认值为 0。
    - **dropout_p** (bool，可选) – dropout概率值，默认值为0.0。
    - **is_causal** (bool，可选) - 是否使用causal模式，默认值：False。
    - **return_softmax** (bool，可选) - 是否返回softmax的结果。默认值：False。
    - **return_softmax_lse** (bool，可选) - 是否返回return_softmax_lse的结果。默认值为 False。
    - **return_seed_offset** (bool，可选) - 是否返回return_seed_offset的结果。默认值为 False。
    - **training** (bool，可选) - 指示是否为训练模式。默认值为 True。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::
`Tensor`，attention的结果。


代码示例
::::::::::::

COPY-FROM: paddle.nn.functional.flash_attention_with_sparse_mask
