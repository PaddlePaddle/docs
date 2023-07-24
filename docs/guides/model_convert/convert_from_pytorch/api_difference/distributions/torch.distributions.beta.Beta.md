## [torch 参数更多 ]torch.distributions.beta.Beta

### [torch.distributions.beta.Beta](https://pytorch.org/docs/stable/distributions.html#torch.distributions.beta.Beta)

```python
torch.distributions.beta.Beta(concentration1, concentration0, validate_args=None)
```

### [paddle.distribution.Beta](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distribution/Beta_cn.html#beta)

```python
paddle.distribution.Beta(alpha, beta)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                     |
| ------- | :----------- | -------------------------------------------------------- |
| concentration1   | alpha            | 表示输入的参数 ，仅参数名不一致。                     |
| concentration0     | beta           | 表示输入的参数，仅参数名不一致。 |
| validate_args     | -            | 是否添加验证环节。Paddle 无此参数，一般对训练结果影响不大，可直接删除。 |
