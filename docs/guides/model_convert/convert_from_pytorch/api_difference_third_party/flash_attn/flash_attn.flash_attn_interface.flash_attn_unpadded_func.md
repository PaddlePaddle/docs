## [torch 参数更多]flash_attn.flash_attn_interface.flash_attn_unpadded_func

### [flash_attn.flash_attn_interface.flash_attn_unpadded_func](https://github.com/Dao-AILab/flash-attention/blob/d0787acc16c3667156b51ce5b01bdafc7594ed39/flash_attn/flash_attn_interface.py#L1050)

```python
flash_attn.flash_attn_interface.flash_attn_unpadded_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,max_seqlen_k, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0， alibi_slopes=None, deterministic=False, return_attn_probs=False, block_table=None)
```

### [paddle.nn.functional.flash_attention.flash_attn_unpadded](https://github.com/PaddlePaddle/Paddle/blob/b32b51b7c21ad62bf794512c849a603c8c0ece44/python/paddle/nn/functional/flash_attention.py#L664)

```python
paddle.nn.functional.flash_attention.flash_attn_unpadded(query, key, value, cu_seqlens_q, cu_seqlens_k,
max_seqlen_q, max_seqlen_k, scale,dropout=0.0, causal=False, return_softmax=False, fixed_seed_offset=None, rng_name='', training=True, name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch           | PaddlePaddle      | 备注                |
| ----------------- | ----------------- | ------------------------------------------------------------ |
| q                 | query             | 注意力模块的 query Tensor。|
| k                 | key               | 注意力模块的 key Tensor。|
| v                 | value             | 注意力模块的 value Tensor。|
| cu_seqlens_q      | cu_seqlens_q      | batch 中累计序列长度，用于索引 q。|
| cu_seqlens_k      | cu_seqlens_k      | batch 中累计序列长度，用于索引 k。|
| max_seqlen_q      | max_seqlen_q      | query 最大序列长度。|
| max_seqlen_k      | max_seqlen_v      | key 最大序列长度。|
| dropout_p         | dropout           | 丢弃概率。    |
| softmax_scale     | scale             | QK^T 的缩放因子。    |
| causal            | causal            | 是否应用因果注意力 mask。  |
| window_size       | -                 | 滑动窗口局部注意力，Paddle 无此参数，暂无转写方式。 |
| softcap           | -                 | 软封顶注意力，Paddle 无此参数，暂无转写方式。    |
| deterministic     | -                 | 是否应用确定性实现，Paddle 无此参数，暂无转写方式。 |
| alibi_slopes      | -                 | 用于注意力得分间的 bias，Paddle 无此参数，暂无转写方式。 |
| block_table       | -                 | block 表，用于 paged KV cache，Paddle 无此参数，暂无转写方式。 |
| return_attn_probs | return_softmax    | 是否返回注意力概率。 |
|                   | fixed_seed_offset | 为 dropout mask 固定 sedd, offset，PyTorch 无此参数，Paddle 保持默认即可。 |
|                   | rng_name          | 选定 rng Generator，PyTorch 无此参数，Paddle 保持默认即可。 |
|                   | training          | 是否在训练阶段，PyTorch 无此参数，Paddle 保持默认即可。 |
|                   | name              | 名称，PyTorch 无此参数，Paddle 保持默认即可。|
