## [ 输入参数类型不一致 ] torch.Tensor.copysign

### [torch.Tensor.copysign](https://pytorch.org/docs/stable/generated/torch.Tensor.copysign.html#torch.Tensor.copysign)

```python
torch.Tensor.copysign(other)
```

### [paddle.Tensor.copysign]()

```python
paddle.Tensor.copysign(y, name=None)
```

其中，PyTorch 与 Paddle 的 `other` 参数所支持类型不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                          |
| ------- | ------------ | ----------------------------- |
| other   | y            | 表示输入的 Tensor ，PyTorch 支持 Python Number 和 Tensor 类型， Paddle 仅支持 Tensor 类型。当输入为 Python Number 类型时，需要转写。  |

### 转写示例
#### other
```python
# PyTorch 写法
result = x.copysign(other=2.)

# Paddle 写法
result = x.copysign(y=paddle.to_tensor(2.))
```
