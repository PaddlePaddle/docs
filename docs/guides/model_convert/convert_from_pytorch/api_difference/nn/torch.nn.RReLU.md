## [torch 参数更多]torch.nn.RReLU

### [torch.nn.RReLU](https://pytorch.org/docs/stable/generated/torch.nn.RReLU.html#torch.nn.RReLU)

```python
torch.nn.RReLU(lower=0.125, upper=0.3333333333333333, inplace=False)
```

### [paddle.nn.RReLU](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/RReLU_cn.html)

```python
paddle.nn.RReLU(lower=1. / 8., upper=1. / 3., name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                                            |
| ------- | ------------ | --------------------------------------------------------------------------------------------------------------- |
| lower   | lower        | 负值斜率的随机值范围下限。                                                                                      |
| upper   | upper        | 负值斜率的随机值范围上限。                                                                                      |
| inplace | -            | 表示在不更改变量的内存地址的情况下，直接修改变量的值，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
