## [ 仅 API 调用方式不一致 ]torch.allclose
### [torch.allclose](https://docs.pytorch.org/docs/stable/generated/torch.allclose.html#torch.allclose)
```python
torch.allclose(input,
               other,
               rtol=1e-05,
               atol=1e-08,
               equal_nan=False)
```

### [paddle.compat.allclose](https://github.com/PaddlePaddle/Paddle/blob/304d5c293907f8620ac6e811097a2847514b863c/python/paddle/compat/__init__.py#L71)
```python
paddle.compat.allclose(input,
               other,
               rtol=1e-05,
               atol=1e-08,
               equal_nan=False)
```

两者功能一致，但调用 API 名称不一致，具体如下：

### 转写示例


```python
# PyTorch 写法
is_close = torch.allclose(a, b)
# Paddle 写法
is_close = paddle.compat.allclose(a, b)
```
