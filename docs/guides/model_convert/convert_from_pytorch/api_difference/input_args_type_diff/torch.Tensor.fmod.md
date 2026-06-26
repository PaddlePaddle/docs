## [ 输入参数类型一致 ]torch.Tensor.fmod
### [torch.Tensor.fmod](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.fmod.html#torch.Tensor.fmod)
```python
torch.Tensor.fmod(other)
```

### [paddle.Tensor.fmod](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#fmod)
```python
paddle.Tensor.fmod(y, name=None)
```

PyTorch 与 Paddle 的 `other` 参数所支持类型一致，均支持 Tensor 和 Python Number 类型。

### 参数映射

| PyTorch | PaddlePaddle | 备注                          |
| ------- | ------------ | ----------------------------- |
| other   | y            | 多维 Tensor 或 Python Number，两者均支持。 |

### 转写示例
#### other 为 Tensor
```python
# PyTorch 写法
result = x.fmod(other=y)

# Paddle 写法
result = x.fmod(y=y)
```

#### other 为标量
```python
# PyTorch 写法
result = x.fmod(other=2.)

# Paddle 写法
result = x.fmod(y=2.)
```
