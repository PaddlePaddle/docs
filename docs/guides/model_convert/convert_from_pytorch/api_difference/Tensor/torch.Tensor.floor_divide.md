## [ 参数不一致 ]torch.Tensor.floor_divide

### [torch.Tensor.floor_divide](https://pytorch.org/docs/stable/generated/torch.Tensor.floor_divide.html?highlight=floor_divide#torch.Tensor.floor_divide)

```python
torch.Tensor.floor_divide(other)
```

### [paddle.Tensor.floor_divide](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#floor-divide-y-name-none)

```python
paddle.Tensor.floor_divide(y, name=None)
```

其中，PyTorch 与 Paddle 的 `other` 参数所支持类型不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                        |
| ------- | ------------ | --------------------------- |
| other   | y            | 多维 Tensor，PyTorch 支持 Tensor 和 Python Number，Paddle 仅支持 Tensor，需要转写。 |

### 转写示例
#### other
```python
# PyTorch 写法
result = x.floor_divide(other=2.)

# Paddle 写法
result = x.floor_divide(y=paddle.to_tensor(2.))
```
