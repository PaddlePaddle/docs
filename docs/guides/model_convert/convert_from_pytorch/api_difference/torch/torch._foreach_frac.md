## [组合替代实现]torch.\_foreach_frac

### [torch.\_foreach_frac](https://pytorch.org/docs/stable/generated/torch._foreach_frac.html#torch-foreach-frac)

```python
torch._foreach_frac(self)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch._foreach_frac(tensors)

# Paddle 写法
[paddle.frac(x) for x in tensors]
```
