## [ 组合替代实现 ]torch.Tensor.narrow_copy

### [torch.Tensor.narrow_copy](https://pytorch.org/docs/stable/generated/torch.Tensor.narrow_copy.html#torch.Tensor.narrow_copy)

```python
torch.Tensor.narrow_copy(dimension, start, length)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = a.narrow_copy(1, 1, 4)

# Paddle 写法
y = paddle.assign(paddle.slice(a, [1], [1], [5]))
```
