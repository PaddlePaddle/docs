## [组合替代实现]torch.\_foreach_frac_

### [torch.\_foreach_frac_](https://pytorch.org/docs/stable/generated/torch._foreach_frac_.html#torch-foreach-frac)

```python
torch._foreach_frac_(self)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch._foreach_frac_(tensors)

# Paddle 写法
[paddle.frac_(x) for x in tensors]
```
