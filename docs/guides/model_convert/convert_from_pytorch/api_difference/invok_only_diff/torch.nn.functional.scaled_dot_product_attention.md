## [ 仅 API 调用方式不一致 ]torch.nn.functional.scaled_dot_product_attention

### [torch.nn.functional.scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
```python
torch.nn.functional.scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
)
```

### [paddle.compat.nn.functional.scaled_dot_product_attention](https://github.com/PaddlePaddle/Paddle/blob/5721d267e434c18fa64ff2b99839c7cb6d4cc04d/python/paddle/compat/nn/functional/sdpa.py#L25)
```python
paddle.compat.nn.functional.scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
)
```

两者功能和用法一致，但 API 路径不一致，只需修改 torch 前缀为 paddle.compat，具体如下：

### 转写示例

```python
# PyTorch 写法
out = torch.nn.functional.scaled_dot_product_attention(query, key, value)

# Paddle 写法
out = paddle.compat.nn.functional.scaled_dot_product_attention(query, key, value)
```
