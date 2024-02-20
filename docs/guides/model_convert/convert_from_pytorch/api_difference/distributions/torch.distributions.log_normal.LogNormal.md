## [torch 参数更多]torch.distributions.log_normal.LogNormal

### [torch.distributions.log_normal.LogNormal](https://pytorch.org/docs/stable/distributions.html#torch.distributions.log_normal.LogNormal)

```python
torch.distributions.log_normal.LogNormal(loc, scale, validate_args=None)
```

### [paddle.distribution.LogNormal](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distribution/LogNormal_cn.html#lognormal)

```python
paddle.distribution.LogNormal(loc, scale, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                    |
| ------------- | ------------ | ----------------------------------------------------------------------- |
| loc           | loc          | 正态分布平均值。                                                        |
| scale         | scale        | 正态分布标准差。                                                        |
| validate_args | -            | 是否添加验证环节。Paddle 无此参数，一般对训练结果影响不大，可直接删除。 |
