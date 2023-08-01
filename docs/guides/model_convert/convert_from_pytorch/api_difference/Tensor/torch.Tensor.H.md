## [ 组合替代实现 ] torch.Tensor.H

### [torch.Tensor.H](https://pytorch.org/docs/stable/tensors.html?#torch.Tensor.H)

```python
torch.Tensor.H
```

Paddle 无此 API，需要组合实现。
PyTorch 中等于 x.transpose(0, 1).conj()，Paddle 中 transpose 参数 perm 为转换后的维度位置

### 转写示例

```python
# Pytorch 写法
y = x.H

# Paddle 写法
y = x.transpose(perm=[1, 0]).conj()
```
