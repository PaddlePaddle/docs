## [ 组合替代实现 ]torch.Tensor.narrow

### [torch.Tensor.narrow](https://pytorch.org/docs/stable/generated/torch.Tensor.narrow.html#torch.Tensor.narrow)

```python
torch.Tensor.narrow(dim, start, length)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = a.narrow(1, 1, 4)

# Paddle 写法
y = paddle.slice(a, [1], [1], [5])
```
