## [ 仅参数名不一致 ]torch.distributions.Distribution.rsample

### [torch.distributions.Distribution.rsample](https://pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution.rsample)

```python
torch.distributions.Distribution.rsample(sample_shape=torch.Size([]))
```

### [paddle.distribution.Distribution.rsample](https://github.com/PaddlePaddle/Paddle/blob/2bbd6f84c1db3e7401732869ee50aef2d9c97bdc/python/paddle/distribution/distribution.py#L96)

```python
paddle.distribution.Distribution.rsample(shape=())
```

其中 PyTorch 和 Paddle 功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注 |
| ------------ | ------------ | ---- |
| sample_shape | shape        | 重参数化的样本形状，仅参数名不同。 |
