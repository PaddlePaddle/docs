## [ torch 参数更多 ]torch.distributions.transforms.PowerTransform

### [torch.distributions.transforms.PowerTransform](https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.SigmoidTransform)

```python
torch.distributions.transforms.PowerTransform(exponent, cache_size=0)
```

### [paddle.distribution.PowerTransform](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distribution/PowerTransform_cn.html#powertransform)

```python
paddle.distribution.PowerTransform(power)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch    | PaddlePaddle | 备注 |
| ---------- | ------------ | -- |
| exponent   | power        | 幂参数。 |
| cache_size | -            | 缓存大小，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
