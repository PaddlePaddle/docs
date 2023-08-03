## [ 组合替代实现 ] torch.Tensor.mT

### [torch.Tensor.mT](https://pytorch.org/docs/stable/tensors.html?#torch.Tensor.mT)

```python
torch.Tensor.mT
```

Paddle 无此 API，需要组合实现。
PyTorch 中等于 x.transpose(-2, -1)，Paddle 中 transpose 参数 perm 为转换后的维度位置。

### 转写示例

```python
# 假设 x 为 4D
# Pytorch 写法
y = x.mT

# Paddle 写法
y = x.transpose(x, perm=[0, 1, 3, 2])
```
