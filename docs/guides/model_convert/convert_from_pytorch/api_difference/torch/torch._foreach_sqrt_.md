## [组合替代实现]torch.\_foreach_sqrt_

### [torch.\_foreach_sqrt_](https://docs.pytorch.org/docs/stable/generated/torch._foreach_sqrt_.html#torch-foreach-sqrt)

```python
torch._foreach_sqrt_(self)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch._foreach_sqrt_(tensors)

# Paddle 写法
[paddle.assign(paddle.sqrt(x), x) for x in tensors]
```
