## [ 组合替代实现 ] torch.Tensor.mT

### [torch.Tensor.mT](https://pytorch.org/docs/stable/tensors.html?#torch.Tensor.mT)

```python
torch.Tensor.mT
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = x.mT

# Paddle 写法
perm = list(range(x.ndim))
perm[-1], perm[-2] = perm[-2], perm[-1]
y = paddle.transpose(x, perm=perm)
```
