## [ 组合替代实现 ]torch.Tensor.addmv

### [torch.Tensor.addmv](https://pytorch.org/docs/stable/generated/torch.Tensor.addmv.html#torch.Tensor.addmv)
```python
torch.Tensor.addmv(mat, vec, beta=1, alpha=1, out=None)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
y = input.addmv(mat, vec, beta=beta, alpha=alpha)

# Paddle 写法
y = beta * input + alpha * paddle.mm(mat, vec)
```
