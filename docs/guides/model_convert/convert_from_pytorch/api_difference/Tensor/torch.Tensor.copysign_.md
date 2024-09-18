## [ 组合替代实现 ] torch.Tensor.copysign_

### [torch.Tensor.copysign_](https://pytorch.org/docs/stable/generated/torch.Tensor.copysign_.html#torch.Tensor.copysign_)

```python
torch.Tensor.copysign_(other)
```

Paddle 无此 API，需要组合实现。

### 转写示例

#### other：输入，类型为 tensor 时
```python
# PyTorch 写法
input.copysign(other=x)

# Paddle 写法
paddle.assign(paddle.copysign(input, x), input)
```

#### other：输入，类型为 number 时
```python
# PyTorch 写法
input.copysign(other=x)

# Paddle 写法
tensor = paddle.to_tensor([x])
paddle.assign(paddle.copysign(input, x), input)
```
