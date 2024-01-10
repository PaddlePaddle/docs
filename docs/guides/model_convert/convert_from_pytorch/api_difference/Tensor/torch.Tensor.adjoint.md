## [ 组合替代实现 ]torch.Tensor.adjoint

### [torch.Tensor.adjoint](https://pytorch.org/docs/stable/generated/torch.Tensor.adjoint.html#torch.Tensor.adjoint)
```python
torch.Tensor.adjoint()
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
y = input.adjoint()

# Paddle 写法
y = paddle.conj(paddle.transpose(input, perm=[0, 2, 1]))

# 注：假设 input 为 3D Tensor， paddle 需要对 input 的后两维转置。
```
