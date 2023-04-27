## [ 组合替代实现 ]torch.Tensor.svd

### [torch.Tensor.svd](https://pytorch.org/docs/stable/generated/torch.Tensor.svd.html#torch.Tensor.svd)

```python
torch.Tensor.svd(some=True, compute_uv=True)
```

Paddle 无此 API，需要组合实现。

### 转写示例
#### some 是否计算完整的 U 和 V 矩阵
```python
# Pytorch 写法
y = a.svd(some=False)

# Paddle 写法
y = paddle.linalg.svd(a, full_matrices=True)
```
