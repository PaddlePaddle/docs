## [组合替代实现] flash_attn.layers.rotary.apply_rotary_emb_func

### [flash_attn.layers.rotary.apply_rotary_emb_fun](https://github.com/Dao-AILab/flash-attention/blob/d0787acc16c3667156b51ce5b01bdafc7594ed39/flash_attn/layers/rotary.py#L94)

```python
flash_attn.layers.rotary.apply_rotary_emb_fun(x,cos,sin,interleaved=False,inplace=False,seqlen_offsets: Union[int, torch.Tensor] = 0,cu_seqlens: Optional[torch.Tensor] = None,max_seqlen: Optional[int] = None,)
```

Paddle 无此 API，需要组合实现。

### 参数映射

| PyTorch           | PaddlePaddle      | 备注                |
| ----------------- | ----------------- | ------------------------------------------------------------ |
| x                 | -                 |  |
| cos               | -                 |  |
| sin               | -                 |  |
| interleaved       | -                 | 为 true 则 GPT-J style |
| inplace           | -                 | 原地操作 |
| seqlen_offsets    | -                 | x 中每个 sequence 的移位 offset |
| cu_seqlens        | -                 |  |
| max_seqlen        | -                 |  |
