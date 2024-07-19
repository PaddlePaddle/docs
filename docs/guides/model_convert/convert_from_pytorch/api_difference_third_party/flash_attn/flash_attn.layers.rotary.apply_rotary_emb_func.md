## [组合替代实现] flash_attn.layers.rotary.apply_rotary_emb_func

### [flash_attn.layers.rotary.apply_rotary_emb_fun](https://github.com/Dao-AILab/flash-attention/blob/d0787acc16c3667156b51ce5b01bdafc7594ed39/flash_attn/layers/rotary.py#L94)

```python
flash_attn.layers.rotary.apply_rotary_emb_fun(x, cos, sin, interleaved=False, inplace=False, seqlen_offsets: Union[int, torch.Tensor] = 0, cu_seqlens: Optional[torch.Tensor] = None, max_seqlen: Optional[int] = None)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
res = flash_attn.layers.rotary.apply_rotary_emb_fun(x, cos, sin)

# Paddle 写法
if not isinstance(cos, paddle.Tensor):
    cos = paddle.to_tensor(cos)
if not isinstance(sin, paddle.Tensor):
    sin = paddle.to_tensor(sin)

def _rotate_half(x):
    from einops import rearrange

    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(axis=-2)
    return paddle.concat((-x2, x1), axis=-1)
# [seq_len,rotary_dim/2] ==>[seq_len, rotary_dim]
cos = paddle.concat([cos,cos],axis=-1)
# [seq_len, rotary_dim] ==>[1,seq_len, 1,rotary_dim]
cos=cos.unsqueeze(axis=1).unsqueeze(axis=0)
# [seq_len,rotary_dim/2] ==>[seq_len, rotary_dim]
sin = paddle.concat([sin,sin],axis=-1)
# [seq_len, rotary_dim] ==>[1,seq_len, 1,rotary_dim]
sin=sin.unsqueeze(axis=1).unsqueeze(axis=0)
t_rot, t_pass = x[..., :cos.shape[-1]], x[..., cos.shape[-1]:]
t_rot = (t_rot * cos) + (_rotate_half(t_rot) * sin)

res = paddle.concat(x=(t_rot, t_pass), axis=-1)
```
