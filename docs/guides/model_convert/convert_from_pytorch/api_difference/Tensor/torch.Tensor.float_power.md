## [组合替代实现]torch.Tensor.float_power

### [torch.Tensor.float_power](https://pytorch.org/docs/stable/generated/torch.Tensor.float_power.html#torch.Tensor.float_power)

```python
torch.Tensor.float_power(exponent)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = torch.range(5).float_power(2)

# Paddle 写法
y = paddle.arange(5).cast(paddle.float64).pow(2)
```
