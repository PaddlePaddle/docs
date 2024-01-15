## [ 组合替代实现 ]torch.Tensor.logaddexp

### [torch.Tensor.logaddexp](https://pytorch.org/docs/stable/generated/torch.Tensor.logaddexp.html#torch.Tensor.logaddexp)

```python
torch.Tensor.logaddexp(other)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
y = a.logaddexp(b)

# Paddle 写法
y = paddle.log(paddle.exp(a) + paddle.exp(b))
```
