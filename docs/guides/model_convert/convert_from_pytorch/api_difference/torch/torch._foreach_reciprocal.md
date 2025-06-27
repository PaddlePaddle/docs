## [组合替代实现]torch.\_foreach_reciprocal

### [torch.\_foreach_reciprocal](https://pytorch.org/docs/stable/generated/torch._foreach_reciprocal.html#torch-foreach-reciprocal)

```python
torch._foreach_reciprocal(self)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch._foreach_reciprocal(tensors)

# Paddle 写法
[paddle.reciprocal(x) for x in tensors]
```
