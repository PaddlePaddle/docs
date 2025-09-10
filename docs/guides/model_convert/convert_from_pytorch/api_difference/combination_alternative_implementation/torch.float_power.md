## [组合替代实现]torch.float_power

### [torch.float_power](https://pytorch.org/docs/stable/generated/torch.float_power.html#torch-float-power)

```python
torch.float_power(input, exponent, *, out=None)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch.float_power(x, y)

# Paddle 写法
paddle.pow(x.cast(paddle.float64), y)
```

#### out：指定输出

```python
# PyTorch 写法
torch.float_power(x, y, out=out)

# Paddle 写法
paddle.assign(paddle.pow(x.cast(paddle.float64), y), out)
```
