## [组合替代实现]torch.Tensor.float_power_

### [torch.Tensor.float_power_](https://pytorch.org/docs/stable/generated/torch.Tensor.float_power_.html#torch.Tensor.float_power_)

```python
torch.Tensor.float_power_(exponent)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = x.float_power_(2)

# Paddle 写法
y = x.cast_(paddle.float64).pow_(2)
```
