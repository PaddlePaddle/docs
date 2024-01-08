## [ 组合替代实现 ]torch.Tensor.reshape_as

### [torch.Tensor.reshape_as](https://pytorch.org/docs/stable/generated/torch.Tensor.reshape_as.html#torch.Tensor.reshape_as)

```python
torch.Tensor.reshape_as(other)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
y = a.reshape_as(b)

# Paddle 写法
y = paddle.reshape(a, b.shape)
```
