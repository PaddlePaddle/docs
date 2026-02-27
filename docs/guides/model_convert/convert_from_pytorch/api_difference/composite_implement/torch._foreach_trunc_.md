## [ 组合替代实现 ]torch.\_foreach_trunc_
### [torch.\_foreach\_trunc\_](https://docs.pytorch.org/docs/stable/generated/torch._foreach_trunc_.html#torch._foreach_trunc_)
```python
torch._foreach_trunc_(self)
```

Paddle 无此 API，需要组合实现。

### 转写示例
```python
# PyTorch 写法
torch._foreach_trunc_(tensors)

# Paddle 写法
[paddle.trunc_(x) for x in tensors]
```
