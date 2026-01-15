## [ torch 参数更多 ]torchvision.models.inception.InceptionE
### [torchvision.models.inception.InceptionE](https://pytorch.org/vision/stable/generated/torchvision.models.inception.InceptionE.html)
```python
torchvision.models.inception.InceptionE(in_channels: int, conv_block: Optional[Callable[..., nn.Module]] = None)
```

### [paddle.vision.models.inceptionv3.InceptionE](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/vision/models/inceptionv3.py)
```python
paddle.vision.models.inceptionv3.InceptionE(num_channels: int)
```

torchvision 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| torchvision       | PaddlePaddle | 备注                                                         |
| :------------ | :----------- | :----------------------------------------------------------- |
| in_channels   | num_channels | 输入通道数，仅参数名不一致。                                 |
| conv_block    | -            | 用于自定义卷积模块，Paddle 无此参数，默认使用 standard conv，可直接删除。 |
