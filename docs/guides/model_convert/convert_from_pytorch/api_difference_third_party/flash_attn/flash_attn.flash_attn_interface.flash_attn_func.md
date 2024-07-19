## [torch 参数更多]flash_attn.flash_attn_interface.flash_attn_func

### [flash_attn.flash_attn_interface.flash_attn_func](https://github.com/Dao-AILab/flash-attention/blob/72e27c6320555a37a83338178caa25a388e46121/flash_attn/flash_attn_interface.py#L808)

```python
flash_attn.flash_attn_interface.flash_attn_func(q, k, v,dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, deterministic=False, return_attn_probs=False)
```

### [paddle.nn.functional.flash_attention.flash_attention](https://github.com/PaddlePaddle/Paddle/blob/900d27c40ef4567d7ea6342f3f0eedd394885ecb/python/paddle/nn/functional/flash_attention.py#L248)

```python
paddle.nn.functional.flash_attention.flash_attention(query, key, value, dropout=0.0, causal=False，return_softmax=False, *, fixed_seed_offset=None, rng_name="", training=True)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch           | PaddlePaddle      | 备注                                                         |
| ----------------- | ----------------- | ------------------------------------------------------------ |
| q                 | query             | 注意力模块的 query Tensor。|
| k                 | key               | 注意力模块的 key Tensor。|
| v                 | value             | 注意力模块的 value Tensor。|
| dropout_p         | dropout           | 丢弃概率。    |
| softmax_scale     | -                 | QK^T 的缩放因子，Paddle 无此参数，暂无转写方式。 |
| causal            | causal            | 是否应用因果注意力 mask。  |
| window_size       | -                 | 滑动窗口局部注意力，Paddle 无此参数，暂无转写方式。 |
| softcap           | -                 | 软封顶注意力，Paddle 无此参数，暂无转写方式。    |
| deterministic     | -                 | 是否应用确定性实现，Paddle 无此参数，暂无转写方式。 |
| alibi_slopes      | -                 | 用于注意力得分间的 bias，Paddle 无此参数，暂无转写方式。|
| return_attn_probs | return_softmax    | 是否返回注意力概率。 |
|                   | fixed_seed_offset | 为 dropout mask 固定 sedd, offset，PyTorch 无此参数，Paddle 保持默认即可。 |
|                   | rng_name          | 选定 rng Generator，PyTorch 无此参数，Paddle 保持默认即可。 |
|                   | training          | 是否在训练阶段，PyTorch 无此参数，Paddle 保持默认即可。 |
