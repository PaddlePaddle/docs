## [ 组合替代实现 ]torch.testing.make_tensor

### [torch.testing.make_tensor](https://pytorch.org/docs/stable/testing.html#torch.testing.make_tensor)

```python
torch.testing.make_tensor(*shape, dtype, device, low=None, high=None, requires_grad=False, noncontiguous=False, exclude_zero=False, memory_format=None)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
x = torch.testing.make_tensor(shape, dtype, device, low=low, high=high, requires_grad=True)

# Paddle 写法
x = paddle.uniform(shape, dtype=dtype, min=low, max=high).to(device)
x.stop_gradient = False
```
