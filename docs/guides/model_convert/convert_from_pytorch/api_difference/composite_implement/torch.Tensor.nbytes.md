## [ 组合替代实现 ]torch.Tensor.nbytes
### [torch.Tensor.nbytes](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.nbytes.html#torch.Tensor.nbytes)
```python
torch.Tensor.nbytes
```

Paddle 无此 API，需要组合实现。

### 转写示例
```python
# PyTorch 写法
y = a.nbytes

# Paddle 写法
y = a.size * a.element_size()
```
