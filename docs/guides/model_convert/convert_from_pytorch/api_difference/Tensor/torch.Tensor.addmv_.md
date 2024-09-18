## [ 组合替代实现 ]torch.Tensor.addmv_

### [torch.Tensor.addmv_](https://pytorch.org/docs/stable/generated/torch.Tensor.addmv_.html#torch.Tensor.addmv_)
```python
torch.Tensor.addmv_(mat, vec, beta=1, alpha=1, out=None)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
input.addmv_(mat, vec, beta=beta, alpha=alpha)

# Paddle 写法
paddle.assign(beta * input + alpha * paddle.mm(mat, vec), input)
```
