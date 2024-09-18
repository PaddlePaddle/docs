## [ 组合替代实现 ]torch.Tensor.addr_

### [torch.Tensor.addr_](https://pytorch.org/docs/stable/generated/torch.Tensor.addr_.html#torch.Tensor.addr_)

```python
torch.Tensor.addr_(vec1, vec2, beta=1, alpha=1)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
y = input.addr_(vec1, vec2, beta=beta, alpha=alpha)

# Paddle 写法
paddle.assign(beta * input + alpha * paddle.outer(vec1, vec2), input)
```
