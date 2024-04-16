## [ 组合替代实现 ]torch.Tensor.addr

### [torch.Tensor.addr](https://pytorch.org/docs/stable/generated/torch.Tensor.addr.html#torch.Tensor.addr)

```python
torch.Tensor.addr(vec1, vec2, beta=1, alpha=1)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
y = input.addr(vec1, vec2, beta=beta, alpha=alpha)

# Paddle 写法
y = beta * input + alpha * paddle.outer(vec1, vec2)
```
