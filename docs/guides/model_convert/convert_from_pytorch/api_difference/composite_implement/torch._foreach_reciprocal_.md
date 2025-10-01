## [组合替代实现]torch.\_foreach_reciprocal_

### [torch.\_foreach_reciprocal_](https://pytorch.org/docs/stable/generated/torch._foreach_reciprocal_.html#torch-foreach-reciprocal)

```python
torch._foreach_reciprocal_(self)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch._foreach_reciprocal_(tensors)

# Paddle 写法
[x.reciprocal_() for x in tensors]
```
