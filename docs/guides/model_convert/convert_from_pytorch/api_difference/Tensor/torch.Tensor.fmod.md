## [ 参数不一致 ]torch.Tensor.fmod

### [torch.Tensor.fmod](https://pytorch.org/docs/stable/generated/torch.Tensor.fmod.html#torch.Tensor.fmod)

```python
torch.Tensor.fmod(other)
```

### [paddle.Tensor.mod](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#mod-y-name-none)

```python
paddle.Tensor.mod(y, name=None)
```

其中，PyTorch 与 Paddle 的 `other` 参数所支持类型不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                          |
| ------- | ------------ | ----------------------------- |
| other   | y            | 多维 Tensor，PyTorch 支持 Tensor 和 Python Number，Paddle 仅支持 Tensor，需要进行转写。 |

### 转写示例
#### other
```python
# PyTorch 写法
result = x.fmod(other=2.)

# Paddle 写法
result = x.mod(y=paddle.to_tensor(2.))
```
