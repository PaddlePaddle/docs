.. _cn_api_paddle_incubate_nn_functional_masked_multihead_attention:

masked_multihead_attention
-------------------------------

.. py:function:: paddle.incubate.nn.functional.masked_multihead_attention(x, cache_kv=None, bias=None, src_mask=None, cum_offsets=None, sequence_lengths=None, rotary_tensor=None, beam_cache_offset=None, qkv_out_scale=None, out_shift=None, out_smooth=None, seq_len=1, rotary_emb_dims=0, use_neox_rotary_style=False, compute_dtype='default', out_scale=- 1, quant_round_type=1, quant_max_bound=127.0, quant_min_bound=- 127.0)

用于文本摘要的蒙版多头注意力机制。

这是一个融合操作符，用于计算 Transformer 模型架构中的蒙版多头注意力。该操作符仅支持在 GPU 上运行。

参数
::::::::::::
    - **x** (Tensor) - 输入张量可以是 2-D 张量。其形状为 [batch_size, 3 * num_head * head_dim]。
    - **cache_kvs** (list(Tensor)|tuple(Tensor)) - 生成模型的缓存结构张量。其形状为 [2, batch_size, num_head, max_seq_len, head_dim]。
    - **bias** (Tensor，可选) - 偏置张量。其形状为 [3, num_head, head_dim]。
    - **src_mask** (Tensor，可选) - 源掩码张量。其形状为 [batch_size, 1, 1, sequence_length]。
    - **sequence_lengths** (Tensor，可选) - 序列长度张量，用于索引输入。其形状为 [batch_size, 1]。
    - **rotary_tensor** (Tensor，可选) - 旋转张量。其数据类型必须为浮点型。其形状为 [batch_size, 1, 1, sequence_length, head_dim]。
    - **beam_cache_offset** (Tensor，可选) - Beam 缓存偏移张量。其形状为 [batch_size, beam_size, max_seq_len + max_dec_len]。
    - **qkv_out_scale** (Tensor，可选) - 量化中使用的 qkv_out_scale 张量。其形状为 [3, num_head, head_dim]。
    - **out_shift** (Tensor，可选) - 量化中使用的 out_shift 张量。
    - **out_smooth** (Tensor，可选) - 量化中使用的 out_smooth 张量。
    - **seq_len** (int，可选) - 序列长度，用于获取输入长度。默认为 1。
    - **rotary_emb_dims** (int，可选) - 旋转嵌入维度。默认为 1。
    - **use_neox_rotary_style** (bool，可选) - 表示是否需要 neox_rotary_style 的标志。默认为 False。
    - **compute_dtype** (string) - 计算数据类型，用于表示输入数据类型。
    - **out_scale** (float，可选) - 量化中使用的 out_scale。默认为 1.0。
    - **quant_round_type** (int，可选) - 量化中使用的 quant_round_type。默认为 1。
    - **quant_max_bound** (float，可选) - 量化中使用的 quant_max_bound。默认为 127.0。
    - **quant_min_bound** (float，可选) - 量化中使用的 quant_min_bound。默认为 -127.0。

返回
::::::::::::
    - Tensor|tuple：如果 "beam_cache_offset_out" 不为 None，则返回元组 (output, cache_kvs_out, beam_cache_offset_out)，其中 output 是蒙版多头注意力层的输出，cache_kvs_out 与输入 `cache_kvs` 原地更新。如果 "beam_cache_offset_out" 为 None，则返回元组 (output, cache_kvs_out)。

形状
::::::::::::
``Tensor``| ``tuple``

代码示例
::::::::::::

COPY-FROM: paddle.incubate.nn.functional.masked_multihead_attention
