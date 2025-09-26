## [组合替代实现]torch.\_foreach_cosh

### [torch.\_foreach_cosh](https://pytorch.org/docs/stable/generated/torch._foreach_cosh.html#torch-foreach-cosh)

```python
torch._foreach_cosh(self)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch._foreach_cosh(tensors)

# Paddle 写法
[paddle.cosh(x) for x in tensors]
```
