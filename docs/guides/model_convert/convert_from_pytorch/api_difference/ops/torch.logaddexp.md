## [ 组合替代实现 ]torch.logaddexp

### [torch.logaddexp](https://pytorch.org/docs/stable/generated/torch.logaddexp.html#torch.logaddexp)

```python
torch.logaddexp(input, other, *, out=None)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
y = torch.logaddexp(a, b)

# Paddle 写法
y = paddle.log(paddle.exp(a) + paddle.exp(b))
```
