## [ 组合替代实现 ]torch.Tensor.adjoint

### [torch.Tensor.adjoint](https://pytorch.org/docs/stable/generated/torch.Tensor.adjoint.html#torch.Tensor.adjoint)
```python
torch.Tensor.adjoint()
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = input.adjoint()

# Paddle 写法
perm = list(range(input.ndim))
perm[-1], perm[-2] = perm[-2], perm[-1]
y = paddle.conj(paddle.transpose(input, perm=perm))
```
