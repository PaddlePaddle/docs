## [组合替代实现]torch.\_foreach_sqrt

### [torch.\_foreach_sqrt](https://pytorch.org/docs/stable/generated/torch._foreach_sqrt.html#torch-foreach-sqrt)

```python
torch._foreach_sqrt(self)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch._foreach_sqrt(tensors)

# Paddle 写法
[paddle.sqrt(x) for x in tensors]
```
