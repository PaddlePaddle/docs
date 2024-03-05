## [ 仅参数名不一致 ]torch.distributions.Distribution.sample

### [torch.distributions.Distribution.sample](https://pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution.sample)

```python
torch.distributions.Distribution.sample(sample_shape=torch.Size([]))
```

### [paddle.distribution.Distribution.sample](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distribution/Distribution_cn.html#sample)

```python
paddle.distribution.Distribution.sample(shape=())
```

其中 PyTorch 和 Paddle 功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注 |
| ------------ | ------------ | -- |
| sample_shape | shape        | 采样形状，仅参数名不一致。 |
