## [ 组合替代实现 ]torch.\_foreach_erfc_
### [torch.\_foreach\_erfc\_](https://docs.pytorch.org/docs/stable/generated/torch._foreach_erfc_.html#torch._foreach_erfc_)
```python
torch._foreach_erfc_(self)
```

Paddle 无此 API，需要组合实现。

### 转写示例
```python
# PyTorch 写法
torch._foreach_erfc_(tensors)

# Paddle 写法
[paddle.assign(1-paddle.erf(x), x) for x in tensors]
```
