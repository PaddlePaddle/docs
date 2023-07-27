## [ 组合替代实现 ] torch.Tensor.mH

### [torch.Tensor.mH](https://pytorch.org/docs/stable/tensors.html?#torch.Tensor.mH)

```python
torch.Tensor.mH
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = x.mH

# Paddle 写法
perm = list(range(x.ndim))
perm[-1], perm[-2] = perm[-2], perm[-1]
y = paddle.conj(paddle.transpose(x, perm=perm))
```
