## [ 组合替代实现 ]torch.\_foreach_floor_
### [torch.\_foreach\_floor\_](https://docs.pytorch.org/docs/stable/generated/torch._foreach_floor_.html#torch._foreach_floor_)
```python
torch._foreach_floor_(self)
```

Paddle 无此 API，需要组合实现。

### 转写示例
```python
# PyTorch 写法
torch._foreach_floor_(tensors)

# Paddle 写法
[x.floor_() for x in tensors]
```
