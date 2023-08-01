## [ 组合替代实现 ] torch.Tensor.mH

### [torch.Tensor.mH](https://pytorch.org/docs/stable/tensors.html?#torch.Tensor.mH)

```python
torch.Tensor.mH
```

Paddle 无此 API，需要组合实现。
PyTorch 中等于 x.transpose(-2, -1).conj()，Paddle 中 transpose 参数 perm 为转换后的维度位置

### 转写示例

```python
# Pytorch 写法
y = x.mH

# Paddle 写法
y = x.transpose(perm=[0, 1, 3, 2]).conj()
```
