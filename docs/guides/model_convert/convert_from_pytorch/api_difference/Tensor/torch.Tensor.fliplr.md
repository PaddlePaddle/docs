## [ 仅 paddle 参数更多 ]torch.Tensor.fliplr

### [torch.Tensor.fliplr](https://pytorch.org/docs/stable/generated/torch.Tensor.fliplr.html?highlight=fliplr#torch.Tensor.fliplr)

```python
Tensor.fliplr()
```

### [paddle.Tensor.flip](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#flip-axis-name-none)

```python
Tensor.flip(axis, name=None)
```

两者功能一致，其中 Paddle 相比 Pytorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                       |
| ------- | ------------ | ---------------------------------------------------------- |
| -       | axis         | 指定进行翻转的轴，Pytorch 无此参数， Paddle 保持默认即可。 |
