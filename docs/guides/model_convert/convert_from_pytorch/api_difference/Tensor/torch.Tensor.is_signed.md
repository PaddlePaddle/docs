## [ 组合替代实现 ]torch.Tensor.is_signed

### [torch.Tensor.is_signed](https://pytorch.org/docs/stable/generated/torch.Tensor.is_signed.html#torch.Tensor.is_signed)

```python
torch.Tensor.is_signed()
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = x.is_signed()

# Paddle 写法
y = x.dtype not in [paddle.uint8]
```
