## [ paddle 参数更多 ]torch.nn.Module.modules

### [torch.nn.Module.modules](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.modules)

```python
torch.nn.Module.modules()
```

### [paddle.nn.Layer.sublayers](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html#sublayers-include-self-false)

```python
paddle.nn.Layer.sublayers(include_self=False)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | -- |
| -       | include_self | 是否包含本层。PyTorch 无此参数，Paddle 保持默认即可。 |
