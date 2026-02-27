## [ 组合替代实现 ]torch.\_foreach_zero_
### [torch.\_foreach\_zero\_](https://docs.pytorch.org/docs/stable/generated/torch._foreach_zero_.html#torch._foreach_zero_)
```python
torch._foreach_zero_(self)
```

Paddle 无此 API，需要组合实现。

### 转写示例
```python
# PyTorch 写法
torch._foreach_zero_(tensors)

# Paddle 写法
[x.zero_() for x in tensors]
```
