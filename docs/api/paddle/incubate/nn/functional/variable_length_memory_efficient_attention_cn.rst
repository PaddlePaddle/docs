.. _cn_api_paddle_incubate_nn_functional_variable_length_memory_efficient_attention:

variable_length_memory_efficient_attention
-------------------------------

.. py:function:: paddle.incubate.nn.functional.variable_length_memory_efficient_attention(query, key, value, seq_lens, kv_seq_lens, mask=None, scale=None, causal=False, pre_cache_length=0)

Cutlass Memory Efficient Variable Attention。
这个方法需要 SM_ARCH 在 sm70, sm75, sm80 中。

参数
::::::::::::

    -  **query** (Tensor) - 查询张量。形状为 [batchsize, seq_len, num_head, head_size].
    - **key** (Tensor) - 关键张量。形状为 [batchsize, seq_len, num_head, head_size].
    - **value** (Tensor) - 值张量。形状为 [batchsize, seq_len, num_head, head_size].
    - **seq_lens** (Tensor) - 批处理中序列的序列长度，用于索引查询。形状为 [batchsize, 1].
    - **kv_seq_lens** (Tensor) - 批处理中序列的序列长度，用于索引键和值。形状为 [batchsize, 1].
    - **mask** (Tensor) - 掩码张量。形状为 [batchsize, 1, query_seq_len, key_seq_len].
    - **scale** (Float) - 注意力矩阵的刻度。默认值为 sqrt (1.0 / head_size).
    - **causal** (Bool) - 是否使用因果掩码。默认值为 False.
    - **pre_cache_length** (Int) - 预缓存的长度。默认值为 0.


返回
::::::::::::
    Tensor: 输出张量。


代码示例
::::::::::::

    COPY-FROM: paddle.callbacks.WandbCallback
