## [ paddle 参数更多 ]torch.nn.LocalResponseNorm
### [torch.nn.LocalResponseNorm](https://pytorch.org/docs/stable/generated/torch.nn.LocalResponseNorm.html?highlight=localre#torch.nn.LocalResponseNorm)

```python
torch.nn.LocalResponseNorm(size,
                           alpha=0.0001,
                           beta=0.75,
                           k=1.0)
```

### [paddle.nn.LocalResponseNorm](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/LocalResponseNorm_cn.html)

```python
paddle.nn.LocalResponseNorm(size,
                            alpha=0.0001,
                            beta=0.75,
                            k=1.0,
                            data_format='NCHW',
                            name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| size   | size | 表示累加的通道数。                   |
| alpha   | alpha | 表示缩放参数。                   |
| beta   | beta | 表示指数。                   |
| k   | k | 表示位移。                   |
| -   | data_format | 指定输入的 format ， PyTorch 无此参数， Paddle 保持默认即可。                  |
