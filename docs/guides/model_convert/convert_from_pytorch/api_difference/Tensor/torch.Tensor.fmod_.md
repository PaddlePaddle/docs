## [ 输入参数类型不一致 ]torch.Tensor.fmod_

### [torch.Tensor.fmod_](https://pytorch.org/docs/stable/generated/torch.Tensor.fmod_.html#torch.Tensor.fmod_)

```python
torch.Tensor.fmod_(other)
```

### [paddle.Tensor.mod_]()

```python
paddle.Tensor.mod_(y, name=None)
```

其中，PyTorch 与 Paddle 的 `other` 参数所支持类型不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                          |
| ------- | ------------ | ----------------------------- |
| other   | y            | 多维 Tensor，PyTorch 支持 Tensor 和 Python Number，Paddle 仅支持 Tensor，需要转写。 |

### 转写示例
#### other
```python
# PyTorch 写法
x.fmod_(other=2.)

# Paddle 写法
x.mod_(y=paddle.to_tensor(2.))
```
