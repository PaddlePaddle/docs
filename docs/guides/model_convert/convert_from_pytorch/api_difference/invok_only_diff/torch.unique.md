## [ 仅 API 调用方式不一致 ]torch.unique

### [torch.unique](https://docs.pytorch.org/docs/stable/generated/torch.unique.html)

```python
torch.unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None)
```

### [paddle.compat.unique](https://docs.pytorch.org/docs/stable/generated/torch.unique.html)

```python
paddle.compat.unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None)
```

两者功能一致，无差异。

### 转写示例

```python
# PyTorch 写法
result = torch.unique(x)

# Paddle 写法
result = paddle.compat.unique(x)
```
