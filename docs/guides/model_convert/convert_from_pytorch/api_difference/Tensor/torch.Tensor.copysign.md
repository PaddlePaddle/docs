## [ 组合替代实现 ] torch.Tensor.copysign

### [torch.Tensor.copysign](https://pytorch.org/docs/stable/generated/torch.Tensor.copysign.html#torch.Tensor.copysign)

```python
torch.Tensor.copysign(other)
```

Paddle 无此 API，需要组合实现。

### 转写示例

#### other：输入，类型为 tensor 时
```python
# PyTorch 写法
y = input.copysign(other=x)

# Paddle 写法
y = paddle.copysign(input, x)
```

#### other：输入，类型为 number 时
```python
# PyTorch 写法
y = input.copysign(other=x)

# Paddle 写法
tensor = paddle.to_tensor([x])
y = paddle.copysign(input, tensor)
```
