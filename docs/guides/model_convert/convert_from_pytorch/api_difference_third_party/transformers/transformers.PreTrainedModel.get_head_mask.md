## [组合替代实现]transformers.PreTrainedModel.get_head_mask

### [transformers.PreTrainedModel.get_head_mask](https://hf-mirror.com/docs/transformers/v4.42.0/en/main_classes/model#transformers.modeling_utils.ModuleUtilsMixin.get_head_mask)

```python
transformers.PreTrainedModel.get_head_mask(head_mask: Optional, num_hidden_layers: int, is_attention_chunked: bool = False)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
head_mask_output = transformers.PreTrainedModel.get_head_mask(head_mask = x, num_hidden_layers, is_attention_chunked)

# Paddle 写法
if x is not None:
    if x.dim() == 1:
        x = x.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        x = x.expand(num_hidden_layers, -1, -1, -1, -1)
    elif x.dim() == 2:
        x = x.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify x for each layer
    assert x.dim() == 5, f"x.dim != 5, instead {x.dim()}"
    head_mask_output = x.to(dtype=paddle.get_default_dtype())  # switch to float if need + fp16 compatibility
    if is_attention_chunked is True:
        head_mask_output = x.unsqueeze(-1)
else:
    head_mask_output = [None] * num_hidden_layers
```
